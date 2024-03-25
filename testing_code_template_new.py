# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/home/pc/test_data"
OUTPUT_CSV_ABSOLUTE_PATH = "/home/pc/output.csv"
# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

def evaluate(file_path):
    # Write your code to predict class for a single audio file instance here
    return predicted_class





def test():
    filenames = []
    predictions = []
    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        prediction = evaluate(absolute_file_name)

        filenames.append(absolute_file_name)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)





# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
# test()
# test_batch()