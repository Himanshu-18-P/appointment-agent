from langchain.tools import tool
import pandas as pd
import dateparser
import os
from dotenv import load_dotenv
from core.utils.vectordb import *
from datetime import datetime

# we need to add human in the loop 

load_dotenv()
indexers = {}

def normalize_date(date_str: str) -> str:
    """
    Convert a natural language date into YYYY-MM-DD format,
    biased toward future dates.

    Args:
        date_str (str): Natural language date input.

    Returns:
        str: Formatted date in YYYY-MM-DD or None.
    """
    dt = dateparser.parse(date_str, settings={"PREFER_DATES_FROM": "future"})
    return dt.strftime("%Y-%m-%d") if dt else None


def normalize_time(time_str: str) -> str:
    """
    Normalize fuzzy time inputs like '9', '9 AM', '0930' into '09:00 AM'.
    If AM/PM is missing, raises an error.

    Args:
        time_str (str): Time input from user.

    Returns:
        str: Time formatted as HH:MM AM/PM.
    """
    if "am" not in time_str.lower() and "pm" not in time_str.lower():
        raise ValueError("⏰ Please specify AM or PM in the time you provided.")
    
    parsed_time = dateparser.parse(time_str)
    if not parsed_time:
        raise ValueError("⏰ Couldn't understand the time format. Please rephrase (e.g., '9:00 AM').")
    
    return parsed_time.strftime("%I:%M %p")


def tools(bot_name: str):
    """
    Factory function that returns a list of LangChain-compatible tools
    for appointment handling. All tools operate on a bot-specific CSV.

    Args:
        bot_name (str): Unique bot folder name.

    Returns:
        List of LangChain tool functions.
    """
    schedule_path = os.path.join("bots_data", bot_name, "schedule.csv")

    def check_availability(date: str, time: str) -> str:
        """
        Check if a specific date and time slot is available for appointment.

        Args:
            date (str): Natural or formatted date string.
            time (str): Time string with AM/PM.

        Returns:
            str: Availability message.
        """
        df = pd.read_csv(schedule_path)
        date = normalize_date(date)
        time = normalize_time(time)
        slot = df[(df["date"] == date) & (df["time"] == time)]

        if slot.empty:
            return "No such slot found."
        elif slot.iloc[0]["is_booked"]:
            return f"{time} on {date} is already booked."
        return f"{time} on {date} is available."

    def book_appointment(date: str, time: str, patient_name: str) -> str:
        """
        Book a free appointment slot for a given date, time, and patient name.

        Args:
            date (str): Natural language or formatted date.
            time (str): Time in AM/PM format.
            patient_name (str): Name of the patient to book.

        Returns:
            str: Confirmation or error message.
        """
        df = pd.read_csv(schedule_path)
        date = normalize_date(date)
        time = normalize_time(time)
        idx = df[(df["date"] == date) & (df["time"] == time)].index

        if len(idx) == 0:
            return "Slot not found."
        if df.loc[idx[0], "is_booked"]:
            return "Slot is already booked."

        df.at[idx[0], "is_booked"] = True
        df.at[idx[0], "patient_name"] = patient_name
        df.to_csv(schedule_path, index=False)

        return f"Appointment booked for {patient_name} at {time} on {date}."

    def list_free_slots(date: str) -> str:
        """
        List all free appointment slots on a given date.
        For today's date, only show future slots (time > now).

        Args:
            date (str): Natural or formatted date string.

        Returns:
            str: List of time slots or message if none are available.
        """
        df = pd.read_csv(schedule_path)
        date = normalize_date(date)

        # Filter only unbooked slots for the given date
        filtered_df = df[(df["date"] == date) & (df["is_booked"] == False)]

        if date == datetime.now().strftime("%Y-%m-%d"):
            now = datetime.now()
            def is_future_time(t):
                try:
                    return datetime.strptime(t, "%I:%M %p").time() > now.time()
                except:
                    return False

            filtered_df = filtered_df[filtered_df["time"].apply(is_future_time)]

        if filtered_df.empty:
            return "No free slots available on that date."

        return "\n".join(f"{row['time']}" for _, row in filtered_df.iterrows())

    def get_datetime(text: str) -> str:
        """
        Parse a natural language string into a full datetime.
        """
        dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
        if not dt:
            return "❌ Could not understand the datetime. Please rephrase."
        return dt.strftime("%Y-%m-%d %I:%M %p")

    # LangChain @tool wrappers
    @tool
    def check_availability_tool(date: str, time: str) -> str:
        """
        Tool: Check availability for a specific date and time slot.
        Requires both date and time with AM/PM.
        """
        return check_availability(date, time)

    @tool
    def book_appointment_tool(date: str, time: str, patient_name: str) -> str:
        """
        Tool: Book an appointment slot for a given date and time with a patient name , ask for name.
        Requires unbooked slot and patient identifier.
        """
        return book_appointment(date, time, patient_name)

    @tool
    def list_free_slots_tool(date: str) -> str:
        """
        Tool: List all unbooked slots for a given date.
        Returns times in 'HH:MM AM/PM' format.
        """
        return list_free_slots(date)

    @tool
    def get_datetime_tool(text: str) -> str:
        """
        Tool: Parse any natural language input (like 'tomorrow at 9') into a standardized datetime.
        Returns a combined string: "YYYY-MM-DD HH:MM AM/PM".
        """
        return get_datetime(text)
    
    @tool
    def reschedule_appointment_tool(patient_name: str) -> str:
        """
        Tool: This tool handles rescheduling requests.
        However, rescheduling is currently not supported in this system.
        """
        return  f"⚠️ Sorry, {patient_name}, we currently do not support rescheduling appointments or cancel. "
    

    @tool
    def context_tool(bot_name: str, user_text: str) -> str:
        """
        Tool: Retrieves the top 5 most relevant context passages 
        for a given query (`user_text`) using the vector store specific to the bot (`bot_name`). 
        Helps the agent answer user queries based on the uploaded PDF content.
        """
        if bot_name not in indexers:
            indexers[bot_name] = PDFIndexer()

        index_dir = os.path.join("vector_store", bot_name)
        results = indexers[bot_name].get_top_k_results(index_dir, user_text)
        print(results)
        return results[0]['text']



    return [
        check_availability_tool,
        book_appointment_tool,
        list_free_slots_tool,
        get_datetime_tool , 
        reschedule_appointment_tool , 
        context_tool
    ]


if __name__ == '__main__':
    print('✅ Tool module is ready.')
