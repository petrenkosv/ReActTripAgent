import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from autogen_core import AgentRuntime, Component, ComponentModel
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    ModelFamily,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel, Field
from typing_extensions import Self

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import ChatAgent, TerminationCondition
from autogen_agentchat.messages import (
    AgentEvent,
    BaseAgentEvent,
    ChatMessage,
    MultiModalMessage,
)
from autogen_agentchat.state import SelectorManagerState
from autogen_agentchat.teams._group_chat._base_group_chat import BaseGroupChat
from autogen_agentchat.teams._group_chat._base_group_chat_manager import (
    BaseGroupChatManager,
)
from autogen_agentchat.teams._group_chat._events import GroupChatTermination

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


class AdvancedSelectorManagerState(SelectorManagerState):
    """State for the AdvancedSelectorGroupChatManager."""

    filtered_messages: Dict[str, List[int]] = Field(default_factory=dict)
    agent_instructions: Dict[str, str] = Field(default_factory=dict)


class AdvancedSelectorGroupChatManager(BaseGroupChatManager):
    """A group chat manager that:
    1. Selects the next speaker using a ChatCompletion model
    2. Selects which messages from history to pass to the agent
    3. Writes instructions to the agent
    """

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        model_client: ChatCompletionClient,
        selector_prompt: str,
        message_filter_prompt: str,
        instruction_prompt: str,
        allow_repeated_speaker: bool,
        selector_func: Optional[
            Callable[[Sequence[AgentEvent | ChatMessage]], str | None]
        ],
        max_selector_attempts: int,
    ) -> None:
        super().__init__(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
        )
        self._model_client = model_client
        self._selector_prompt = selector_prompt
        self._message_filter_prompt = message_filter_prompt
        self._instruction_prompt = instruction_prompt
        self._previous_speaker: Optional[str] = None
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func
        self._max_selector_attempts = max_selector_attempts
        self._filtered_messages: Dict[
            str, List[int]
        ] = {}  # Agent name -> list of message indices
        self._agent_instructions: Dict[str, str] = {}  # Agent name -> instruction

    async def validate_group_state(self, messages: List[ChatMessage] | None) -> None:
        pass

    async def reset(self) -> None:
        self._current_turn = 0
        self._message_thread.clear()
        if self._termination_condition is not None:
            await self._termination_condition.reset()
        self._previous_speaker = None
        self._filtered_messages.clear()
        self._agent_instructions.clear()

    async def save_state(self) -> Mapping[str, Any]:
        state = AdvancedSelectorManagerState(
            message_thread=list(self._message_thread),
            current_turn=self._current_turn,
            previous_speaker=self._previous_speaker,
            filtered_messages=self._filtered_messages,
            agent_instructions=self._agent_instructions,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        selector_state = AdvancedSelectorManagerState.model_validate(state)
        self._message_thread = list(selector_state.message_thread)
        self._current_turn = selector_state.current_turn
        self._previous_speaker = selector_state.previous_speaker
        self._filtered_messages = selector_state.filtered_messages
        self._agent_instructions = selector_state.agent_instructions

    async def select_speaker(self, thread: List[AgentEvent | ChatMessage]) -> str:
        """Selects the next speaker in a group chat using a ChatCompletion client,
        with the selector function as override if it returns a speaker name.
        """

        # Use the selector function if provided.
        if self._selector_func is not None:
            speaker = self._selector_func(thread)
            if speaker is not None:
                if speaker not in self._participant_names:
                    raise ValueError(
                        f"Selector function returned an invalid speaker name: {speaker}. "
                        f"Expected one of: {self._participant_names}."
                    )
                # Skip the model based selection.
                return speaker

        # Construct the history of the conversation.
        history_messages: List[str] = []
        for msg in thread:
            if isinstance(msg, BaseAgentEvent):
                # Ignore agent events.
                continue
            message = f"{msg.source}:"
            if isinstance(msg.content, str):
                message += f" {msg.content}"
            elif isinstance(msg, MultiModalMessage):
                for item in msg.content:
                    if isinstance(item, str):
                        message += f" {item}"
                    else:
                        message += " [Image]"
            else:
                raise ValueError(f"Unexpected message type in selector: {type(msg)}")
            history_messages.append(message.rstrip() + "\n\n")
        history = "\n".join(history_messages)

        # Construct agent roles.
        roles = ""
        for topic_type, description in zip(
            self._participant_names, self._participant_descriptions, strict=True
        ):
            roles += re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
        roles = roles.strip()

        # Construct the candidate agent list to be selected from, skip the previous speaker if not allowed.
        if self._previous_speaker is not None and not self._allow_repeated_speaker:
            participants = [
                p for p in self._participant_names if p != self._previous_speaker
            ]
        else:
            participants = list(self._participant_names)
        assert len(participants) > 0

        # Select the next speaker.
        if len(participants) > 1:
            agent_name = await self._select_speaker(
                roles, participants, history, self._max_selector_attempts
            )
        else:
            agent_name = participants[0]

        # After selecting the speaker, filter messages and generate instructions
        await self._filter_messages_for_agent(agent_name, thread)
        await self._generate_instructions_for_agent(agent_name, thread)

        self._previous_speaker = agent_name
        trace_logger.debug(f"Selected speaker: {agent_name}")
        return agent_name

    async def _select_speaker(
        self, roles: str, participants: List[str], history: str, max_attempts: int
    ) -> str:
        select_speaker_prompt = self._selector_prompt.format(
            roles=roles, participants=str(participants), history=history
        )
        select_speaker_messages: List[SystemMessage | UserMessage | AssistantMessage]
        if ModelFamily.is_openai(self._model_client.model_info["family"]):
            select_speaker_messages = [SystemMessage(content=select_speaker_prompt)]
        else:
            # Many other models need a UserMessage to respond to
            select_speaker_messages = [
                UserMessage(content=select_speaker_prompt, source="user")
            ]

        num_attempts = 0
        while num_attempts < max_attempts:
            num_attempts += 1
            response = await self._model_client.create(messages=select_speaker_messages)
            assert isinstance(response.content, str)
            select_speaker_messages.append(
                AssistantMessage(content=response.content, source="selector")
            )
            # NOTE: we use all participant names to check for mentions, even if the previous speaker is not allowed.
            # This is because the model may still select the previous speaker, and we want to catch that.
            mentions = self._mentioned_agents(response.content, self._participant_names)
            if len(mentions) == 0:
                trace_logger.debug(
                    f"Model failed to select a valid name: {response.content} (attempt {num_attempts})"
                )
                feedback = f"No valid name was mentioned. Please select from: {str(participants)}."
                select_speaker_messages.append(
                    UserMessage(content=feedback, source="user")
                )
            elif len(mentions) > 1:
                trace_logger.debug(
                    f"Model selected multiple names: {str(mentions)} (attempt {num_attempts})"
                )
                feedback = f"Expected exactly one name to be mentioned. Please select only one from: {str(participants)}."
                select_speaker_messages.append(
                    UserMessage(content=feedback, source="user")
                )
            else:
                agent_name = list(mentions.keys())[0]
                if (
                    not self._allow_repeated_speaker
                    and self._previous_speaker is not None
                    and agent_name == self._previous_speaker
                ):
                    trace_logger.debug(
                        f"Model selected the previous speaker: {agent_name} (attempt {num_attempts})"
                    )
                    feedback = f"Repeated speaker is not allowed, please select a different name from: {str(participants)}."
                    select_speaker_messages.append(
                        UserMessage(content=feedback, source="user")
                    )
                else:
                    # Valid selection
                    trace_logger.debug(
                        f"Model selected a valid name: {agent_name} (attempt {num_attempts})"
                    )
                    return agent_name

        if self._previous_speaker is not None:
            trace_logger.warning(
                f"Model failed to select a speaker after {max_attempts}, using the previous speaker."
            )
            return self._previous_speaker
        trace_logger.warning(
            f"Model failed to select a speaker after {max_attempts} and there was no previous speaker, using the first participant."
        )
        return participants[0]

    async def _filter_messages_for_agent(
        self, agent_name: str, thread: List[AgentEvent | ChatMessage]
    ) -> None:
        """Filters the messages that should be passed to the agent."""
        # Construct the history of the conversation with indices
        history_messages: List[Tuple[int, str]] = []
        message_index = 0
        for msg in thread:
            if isinstance(msg, BaseAgentEvent):
                # Ignore agent events.
                continue
            message = f"{msg.source}:"
            if isinstance(msg.content, str):
                message += f" {msg.content}"
            elif isinstance(msg, MultiModalMessage):
                for item in msg.content:
                    if isinstance(item, str):
                        message += f" {item}"
                    else:
                        message += " [Image]"
            else:
                raise ValueError(
                    f"Unexpected message type in message filter: {type(msg)}"
                )
            history_messages.append((message_index, message.rstrip()))
            message_index += 1

        # Format the history with indices
        indexed_history = "\n\n".join(
            [f"[{idx}] {msg}" for idx, msg in history_messages]
        )

        # Construct the filter prompt
        filter_prompt = self._message_filter_prompt.format(
            agent_name=agent_name, history=indexed_history
        )

        filter_messages: List[SystemMessage | UserMessage | AssistantMessage]
        if ModelFamily.is_openai(self._model_client.model_info["family"]):
            filter_messages = [SystemMessage(content=filter_prompt)]
        else:
            filter_messages = [UserMessage(content=filter_prompt, source="user")]

        # Get the filtered message indices
        response = await self._model_client.create(messages=filter_messages)
        assert isinstance(response.content, str)

        # Parse the response to get the message indices
        # The response should be a list of indices, e.g., "[0, 2, 5]" or "0, 2, 5"
        try:
            # Try to extract a list of numbers from the response
            indices_str = re.search(r"\[([0-9, ]+)\]", response.content)
            if indices_str:
                indices = [
                    int(idx.strip())
                    for idx in indices_str.group(1).split(",")
                    if idx.strip()
                ]
            else:
                # If no list format is found, try to extract comma-separated numbers
                indices = [
                    int(idx.strip()) for idx in re.findall(r"\b\d+\b", response.content)
                ]

            # Store the filtered message indices for this agent
            self._filtered_messages[agent_name] = indices
            trace_logger.debug(f"Filtered messages for {agent_name}: {indices}")
        except Exception as e:
            trace_logger.warning(
                f"Failed to parse message filter response: {response.content}. Error: {e}"
            )
            # If parsing fails, include all messages
            self._filtered_messages[agent_name] = list(range(len(history_messages)))

    async def _generate_instructions_for_agent(
        self, agent_name: str, thread: List[AgentEvent | ChatMessage]
    ) -> None:
        """Generates instructions for the agent."""
        # Construct the history of the conversation
        history_messages: List[str] = []
        for msg in thread:
            if isinstance(msg, BaseAgentEvent):
                # Ignore agent events.
                continue
            message = f"{msg.source}:"
            if isinstance(msg.content, str):
                message += f" {msg.content}"
            elif isinstance(msg, MultiModalMessage):
                for item in msg.content:
                    if isinstance(item, str):
                        message += f" {item}"
                    else:
                        message += " [Image]"
            else:
                raise ValueError(
                    f"Unexpected message type in instruction generator: {type(msg)}"
                )
            history_messages.append(message.rstrip())

        history = "\n\n".join(history_messages)

        # Get the filtered messages if available
        filtered_indices = self._filtered_messages.get(
            agent_name, list(range(len(history_messages)))
        )
        filtered_history = "\n\n".join(
            [
                history_messages[idx]
                for idx in filtered_indices
                if idx < len(history_messages)
            ]
        )

        # Construct the instruction prompt
        instruction_prompt = self._instruction_prompt.format(
            agent_name=agent_name, history=history, filtered_history=filtered_history
        )

        instruction_messages: List[SystemMessage | UserMessage | AssistantMessage]
        if ModelFamily.is_openai(self._model_client.model_info["family"]):
            instruction_messages = [SystemMessage(content=instruction_prompt)]
        else:
            instruction_messages = [
                UserMessage(content=instruction_prompt, source="user")
            ]

        # Generate the instructions
        response = await self._model_client.create(messages=instruction_messages)
        assert isinstance(response.content, str)

        # Store the instructions for this agent
        self._agent_instructions[agent_name] = response.content
        trace_logger.debug(
            f"Generated instructions for {agent_name}: {response.content}"
        )

    def _mentioned_agents(
        self, message_content: str, agent_names: List[str]
    ) -> Dict[str, int]:
        """Counts the number of times each agent is mentioned in the provided message content.
        Agent names will match under any of the following conditions (all case-sensitive):
        - Exact name match
        - If the agent name has underscores it will match with spaces instead (e.g. 'Story_writer' == 'Story writer')
        - If the agent name has underscores it will match with '\\_' instead of '_' (e.g. 'Story_writer' == 'Story\\_writer')
        """
        mentions: Dict[str, int] = dict()
        for name in agent_names:
            # Finds agent mentions, taking word boundaries into account,
            # accommodates escaping underscores and underscores as spaces
            regex = (
                r"(?<=\W)("
                + re.escape(name)
                + r"|"
                + re.escape(name.replace("_", " "))
                + r"|"
                + re.escape(name.replace("_", r"\_"))
                + r")(?=\W)"
            )
            # Pad the message to help with matching
            count = len(re.findall(regex, f" {message_content} "))
            if count > 0:
                mentions[name] = count
        return mentions

    async def get_filtered_messages(self, agent_name: str) -> List[int]:
        """Returns the filtered message indices for the given agent."""
        return self._filtered_messages.get(agent_name, [])

    async def get_agent_instructions(self, agent_name: str) -> str:
        """Returns the instructions for the given agent."""
        return self._agent_instructions.get(agent_name, "")


class AdvancedSelectorGroupChatConfig(BaseModel):
    """The declarative configuration for AdvancedSelectorGroupChat."""

    participants: List[ComponentModel]
    model_client: ComponentModel
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    selector_prompt: str
    message_filter_prompt: str
    instruction_prompt: str
    allow_repeated_speaker: bool
    max_selector_attempts: int = 3


class AdvancedSelectorGroupChat(
    BaseGroupChat, Component[AdvancedSelectorGroupChatConfig]
):
    """A group chat team that:
    1. Selects the next speaker from a group of agents
    2. Selects what messages from history of dialog to pass to the agent
    3. Writes instructions to the agent

    Args:
        participants (List[ChatAgent]): The participants in the group chat,
            must have unique names and at least two participants.
        model_client (ChatCompletionClient): The ChatCompletion model client used
            for selection and instruction generation.
        termination_condition (TerminationCondition, optional): The termination condition for the group chat. Defaults to None.
            Without a termination condition, the group chat will run indefinitely.
        max_turns (int, optional): The maximum number of turns in the group chat before stopping. Defaults to None, meaning no limit.
        selector_prompt (str, optional): The prompt template to use for selecting the next speaker.
            Available fields: '{roles}', '{participants}', and '{history}'.
        message_filter_prompt (str, optional): The prompt template to use for filtering messages.
            Available fields: '{agent_name}' and '{history}'.
        instruction_prompt (str, optional): The prompt template to use for generating instructions.
            Available fields: '{agent_name}', '{history}', and '{filtered_history}'.
        allow_repeated_speaker (bool, optional): Whether to include the previous speaker in the list of candidates to be selected for the next turn.
            Defaults to False.
        max_selector_attempts (int, optional): The maximum number of attempts to select a speaker using the model. Defaults to 3.
        selector_func (Callable[[Sequence[AgentEvent | ChatMessage]], str | None], optional): A custom selector
            function that takes the conversation history and returns the name of the next speaker.
            If provided, this function will be used to override the model to select the next speaker.
            If the function returns None, the model will be used to select the next speaker.

    Raises:
        ValueError: If the number of participants is less than two or if the prompts are invalid.
    """

    component_config_schema = AdvancedSelectorGroupChatConfig
    component_provider_override = (
        "swarm2.advanced_selector_group_chat.AdvancedSelectorGroupChat"
    )

    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        selector_prompt: str = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
""",
        message_filter_prompt: str = """You are a message filter for a group chat. Your task is to select which messages from the conversation history should be passed to the next speaker.

The next speaker is: {agent_name}

Below is the conversation history with message indices:
{history}

Select the message indices that are most relevant for {agent_name} to see in order to generate an appropriate response.
Return only a list of indices in the format [0, 1, 2, ...] or as comma-separated numbers like "0, 1, 2, ...".
""",
        instruction_prompt: str = """You are an instruction writer for a group chat. Your task is to write clear instructions for the next speaker based on the conversation history.

The next speaker is: {agent_name}

Below is the full conversation history:
{history}

Below are the filtered messages that will be shown to {agent_name}:
{filtered_history}

Write specific instructions for {agent_name} on how they should respond to the conversation.
Focus on what {agent_name} should address, what tone they should use, and any specific points they should include in their response.
""",
        allow_repeated_speaker: bool = False,
        max_selector_attempts: int = 3,
        selector_func: Callable[[Sequence[AgentEvent | ChatMessage]], str | None]
        | None = None,
    ):
        super().__init__(
            participants,
            group_chat_manager_name="AdvancedSelectorGroupChatManager",
            group_chat_manager_class=AdvancedSelectorGroupChatManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
        )
        # Validate the participants.
        if len(participants) < 2:
            raise ValueError(
                "At least two participants are required for AdvancedSelectorGroupChat."
            )
        self._selector_prompt = selector_prompt
        self._message_filter_prompt = message_filter_prompt
        self._instruction_prompt = instruction_prompt
        self._model_client = model_client
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func
        self._max_selector_attempts = max_selector_attempts

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
    ) -> Callable[[], BaseGroupChatManager]:
        return lambda: AdvancedSelectorGroupChatManager(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
            self._model_client,
            self._selector_prompt,
            self._message_filter_prompt,
            self._instruction_prompt,
            self._allow_repeated_speaker,
            self._selector_func,
            self._max_selector_attempts,
        )

    def _to_config(self) -> AdvancedSelectorGroupChatConfig:
        return AdvancedSelectorGroupChatConfig(
            participants=[
                participant.dump_component() for participant in self._participants
            ],
            model_client=self._model_client.dump_component(),
            termination_condition=self._termination_condition.dump_component()
            if self._termination_condition
            else None,
            max_turns=self._max_turns,
            selector_prompt=self._selector_prompt,
            message_filter_prompt=self._message_filter_prompt,
            instruction_prompt=self._instruction_prompt,
            allow_repeated_speaker=self._allow_repeated_speaker,
            max_selector_attempts=self._max_selector_attempts,
        )

    @classmethod
    def _from_config(cls, config: AdvancedSelectorGroupChatConfig) -> Self:
        return cls(
            participants=[
                BaseChatAgent.load_component(participant)
                for participant in config.participants
            ],
            model_client=ChatCompletionClient.load_component(config.model_client),
            termination_condition=TerminationCondition.load_component(
                config.termination_condition
            )
            if config.termination_condition
            else None,
            max_turns=config.max_turns,
            selector_prompt=config.selector_prompt,
            message_filter_prompt=config.message_filter_prompt,
            instruction_prompt=config.instruction_prompt,
            allow_repeated_speaker=config.allow_repeated_speaker,
            max_selector_attempts=config.max_selector_attempts,
        )
