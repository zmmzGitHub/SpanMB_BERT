# split result file by document. Generate .a2 file for evaluating
# a = ['1test\n', '2test\n']
# with open("test.txt", "w") as f:
#     f.writelines(a)
# with open("test.txt", "a") as f:  # append from \n of last input
#     f.write("3test")
import os
from spanmb.data.dataset_readers import document


def convert_to_evaluation_format(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    docs = document.Dataset.from_jsonl(input_file)
    docs.output_predict(output_dir)


def main():
    fold_name = ["dev"]   # , "test"
    for fold in fold_name:
        input_file = f"models/bacteria/test/{fold}_file_result.json"
        output_dir = f"models/bacteria/test/{fold}_output"
        convert_to_evaluation_format(input_file, output_dir)


if __name__ == "__main__":
    main()
