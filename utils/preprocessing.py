import pandas as pd
from sklearn.preprocessing import LabelEncoder

class UserLabelProcessor:
    def __init__(self, input_csv_path, user_csv_path, output_csv_path):
        self.input_csv_path = input_csv_path
        self.user_csv_path = user_csv_path
        self.output_csv_path = output_csv_path

    def melt_and_save(self):
        # Read user information from the CSV file
        df = pd.read_csv(self.input_csv_path)

        # Melt the DataFrame to create the rating DataFrame
        label_df = pd.melt(df, id_vars=["user_id"], value_vars=["금융", "증권", "부동산", "글로벌경제", "생활경제", "경제일반"],
                          var_name="label", value_name="rating")
        
        # read user information
        user = pd.read_csv(self.user_csv_path)
        # drop columns
        user = user.drop(["금융", "증권", "부동산", "글로벌경제", "생활경제", "경제일반"], axis=1)
        # label encoding for categorical columns
        cat_col = ["gender", "age", "occupation", "address"]        
        for cat in cat_col:
            encoder = LabelEncoder()
            user[cat] = encoder.fit_transform(user[cat])
        
        label_df = pd.merge(label_df, user)

        # Save the rating DataFrame to another CSV file
        label_df.to_csv(self.output_csv_path, index=False)