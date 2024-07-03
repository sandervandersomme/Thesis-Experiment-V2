import xml.etree.ElementTree as ET
import pandas as pd
import pm4py

# def parse_xes(file_path):
#     tree = ET.parse(file_path)
#     root = tree.getroot()
    
#     # List to store event data
#     data = []

#     # Iterate over traces and events in the log
#     for trace in root.findall('.//trace'):
#         trace_id = None
#         # Find the trace ID (usually stored in a string element with key 'concept:name')
#         for string in trace.findall('string'):
#             if string.get('key') == 'concept:name':
#                 trace_id = string.get('value')
#                 break
        
#         # Iterate over events within each trace
#         for event in trace.findall('event'):
#             event_data = {'trace_id': trace_id}  # Include trace ID in event data
#             # Extract string elements from the event
#             for string in event.findall('string'):
#                 key = string.get('key')
#                 value = string.get('value')
#                 event_data[key] = value
#             # Extract date elements from the event
#             for date in event.findall('date'):
#                 key = date.get('key')
#                 value = date.get('value')
#                 event_data[key] = value
#             # Extract integer elements from the event
#             for integer in event.findall('int'):
#                 key = integer.get('key')
#                 value = integer.get('value')
#                 event_data[key] = value
#             data.append(event_data)
    
#     # Convert list of event data to DataFrame
#     df = pd.DataFrame(data)
#     return df

def save_to_csv(df: pd.DataFrame, output_file_path: str):
    df.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")


if __name__ == "__main__":
    input_path = "datasets/raw/sepsis/sepsis.xes"
    output_path = "datasets/raw/sepsis/sepsis_raw.csv"
    
    log = pm4py.read_xes(input_path) #Input Filename
    df = pm4py.convert_to_dataframe(log)
    # df = parse_xes(input_path)
    save_to_csv(df, output_path)
