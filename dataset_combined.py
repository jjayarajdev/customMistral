import pandas as pd

def merge_jsonl_with_pandas(file_paths, output_file):
    dfs = [pd.read_json(path, lines=True) for path in file_paths]
    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.to_json(output_file, orient='records', lines=True)

# Example usage
file_paths = ['data.jsonl', 'data_new.jsonl']
output_file = 'dataset.jsonl'
merge_jsonl_with_pandas(file_paths, output_file)
