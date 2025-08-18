import os
import json
import pandas as pd

import os
import json
import pandas as pd

class HandleData:
    def __init__(self):
        self.BASE_DIR = "bots_data"

    def get_bot_folder(self, bot_name: str) -> str:
        return os.path.join(self.BASE_DIR, bot_name)
    
    def get_schedule_path(self, bot_name: str) -> str:
        return os.path.join(self.get_bot_folder(bot_name), "schedule.csv")

    def get_meta_path(self, bot_name: str) -> str:
        return os.path.join(self.get_bot_folder(bot_name), "meta.json")
    
    def savejson(self, bot_name: str, bot_data: dict) -> str:
        bot_folder = self.get_bot_folder(bot_name)
        os.makedirs(bot_folder, exist_ok=True)  # ✅ Create full bot folder path

        # ✅ Save empty schedule
        schedule_path = self.get_schedule_path(bot_name)
        pd.DataFrame(columns=["date", "time", "is_booked", "patient_name"]).to_csv(schedule_path, index=False)

        # ✅ Save metadata
        with open(self.get_meta_path(bot_name), "w") as f:
            json.dump(bot_data, f, indent=2)

        return "data saved"



if __name__ == '__main__':
    print('done')