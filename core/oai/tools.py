from langchain.tools import tool
import pandas as pd
import dateparser
from dotenv import load_dotenv


load_dotenv()

# Load the schedule CSV into memory
schedule_df = pd.read_csv("schedule.csv")

# ----------- Normalizers ------------
def normalize_date(date_str: str) -> str:
    """
    Convert a natural language date into YYYY-MM-DD format,
    biased toward future dates.
    """
    dt = dateparser.parse(date_str, settings={"PREFER_DATES_FROM": "future"})
    return dt.strftime("%Y-%m-%d") if dt else None

def normalize_time(time_str: str) -> str:
    """
    Normalize fuzzy time inputs like '9', '9 AM', '0930' into '09:00 AM'.
    If AM/PM is missing, ask the LLM to include it in the request.
    """
    if "am" not in time_str.lower() and "pm" not in time_str.lower():
        raise ValueError("⏰ Please specify AM or PM in the time you provided.")
    parsed_time = dateparser.parse(time_str)
    if not parsed_time:
        raise ValueError("⏰ Couldn't understand the time format. Please rephrase (e.g., '9:00 AM').")
    return parsed_time.strftime("%I:%M %p")

# ----------- Appointment Logic ------------
def check_availability(date: str, time: str) -> str:
    """
    Check if a specific date and time slot is available for appointment.

    Args:
        date (str): Natural or formatted date string.
        time (str): Time string with AM/PM.

    Returns:
        str: Availability message.
    """
    global schedule_df
    date = normalize_date(date)
    time = normalize_time(time)
    slot = schedule_df[(schedule_df["date"] == date) & (schedule_df["time"] == time)]

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
    global schedule_df
    date = normalize_date(date)
    time = normalize_time(time)
    idx = schedule_df[(schedule_df["date"] == date) & (schedule_df["time"] == time)].index

    if len(idx) == 0:
        return "Slot not found."
    if schedule_df.loc[idx[0], "is_booked"]:
        return "Slot is already booked."

    schedule_df.at[idx[0], "is_booked"] = True
    schedule_df.at[idx[0], "patient_name"] = patient_name
    schedule_df.to_csv("schedule.csv", index=False)

    return f"Appointment booked for {patient_name} at {time} on {date}."

def list_free_slots(date: str) -> str:
    """
    List all free appointment slots on a given date.

    Args:
        date (str): Natural or formatted date string.

    Returns:
        str: List of time slots or message if none are available.
    """
    global schedule_df
    date = normalize_date(date)
    free_slots = schedule_df[(schedule_df["date"] == date) & (schedule_df["is_booked"] == False)]

    if free_slots.empty:
        return "No free slots available on that date."

    return "\n".join(f"{row['time']}" for _, row in free_slots.iterrows())

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
    Tool: Book an appointment slot for a given date and time with a patient name.
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
    dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
    if not dt:
        return "❌ Could not understand the datetime. Please rephrase."
    return dt.strftime("%Y-%m-%d %I:%M %p")

tools = [check_availability_tool, book_appointment_tool, list_free_slots_tool, get_datetime_tool]


if __name__ == '__main__':
    print('done')