import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from aiogram import Bot

from .agent import weather_agent, summary_agent, sights_agent, finance_agent
from .config import TELEGRAM_BOT_TOKEN

# Initialize bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Initialize FastAPI app
app = FastAPI()

class TaskRequest(BaseModel):
    task: str
    user_id: int
    console_mode: bool = False

async def send_telegram_message(user_id, message):
    """
    Send a message to a Telegram user by user ID.

    Args:
        user_id (int): Telegram user ID
        message (str): Message to send
    """
    try:
        await bot.send_message(chat_id=user_id, text=message)
        logging.info(f"Message sent to user {user_id}")
    except Exception as e:
        logging.error(f"Failed to send message to user {user_id}: {e}")

async def main(task: str, user_id: int, console_mode: bool):
    """
    Run the agent chat and send the final message to the user via Telegram if user_id is provided.

    Args:
        task (str): The task description for the agents
        user_id (int, optional): Telegram user ID to send the final message to
        console_mode (bool, optional): If True, run in console mode
    """
    # Define termination condition based on text mention
    termination = TextMentionTermination("TERMINATE")
    
    # Initialize group chat with agents and termination condition
    group_chat = RoundRobinGroupChat(
        [weather_agent, sights_agent, finance_agent, summary_agent],
        termination_condition=termination,
    )
    
    if not console_mode:
        # Run the group chat with the provided task
        result = await group_chat.run(task=task)
        final_message = result.messages[-1].content.replace("TERMINATE", "")
        
        # Send the final message to the user via Telegram if user_id is provided
        if user_id:
            await send_telegram_message(user_id, final_message)
            logging.info(f"Final message sent to user {user_id}")
        else:
            logging.info("No user_id provided, skipping Telegram message")
            print(final_message)
        
        return final_message

    # Run the group chat in console mode
    await Console(
        group_chat.run_stream(
            task="Plan a trip to Berlin 3 days. I`ll have meeting from 9am to 16pm every day exept last"
        )
    )

@app.post("/execute_task")
async def execute_task(request: TaskRequest):
    try:
        result = await main(request.task, request.user_id, request.console_mode)
        return {"status": "success", "result": result}
    except Exception as e:
        logging.error(f"Error executing task: {e}")
        raise HTTPException(status_code=500, detail="Error executing task")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
