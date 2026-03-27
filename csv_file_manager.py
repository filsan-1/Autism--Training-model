# csv_file_manager.py

# This script manages CSV files in a beginner-friendly manner.

import csv

# Opening a CSV file
# (Messy section: the try-except is not properly handling exceptions)
try:
    with open('data.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Debugging trail: let's print each row to see what's inside!
            print(row)  # <-- Remove this in production!
except Exception as e:
    print(f'Error opening file: {e}')  # Not very informative

# Function to write data to CSV

def write_to_csv(data):
    # Messy section: not checking if the data is valid
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            # Debugging trail: print each item being written to debug
            print(f'Writing item: {item}')  # <-- Remove this in production!
            writer.writerow(item)

# Example usage
if __name__ == '__main__':
    data = [['name', 'age'], ['Alice', '30'], ['Bob', '25']]
    write_to_csv(data)
    # Add more beginner comments here to explain each step more clearly

# Note: This code contains messy sections intentionally to illustrate debugging and errors.