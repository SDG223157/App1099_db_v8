import re
import ast
import json

# Read the content from tickers.ts with correct encoding
with open('tickers.ts', 'r', encoding='utf-8') as file:
    content = file.read()

# Find the position of the array within the content
start = content.find('export const tickers = [') + len('export const tickers = [')
end = content.rfind('];') + 1
array_content = content[start:end].strip()

# Use regex to find all ticker objects
pattern = re.compile(r'{[^{}]*}', re.DOTALL)
matches = pattern.findall(array_content)

# List to hold the ticker dictionaries
tickers_list = []

# Process each matched ticker object
for match in matches:
    try:
        # Replace unquoted keys with quoted ones
        match = match.replace(' symbol: ', "'symbol': ").replace(' name: ', "'name': ")
        # Remove the trailing comma if present
        if match.endswith(','):
            match = match[:-1]
        # Parse the modified string into a dictionary
        ticker = ast.literal_eval(match)
        tickers_list.append(ticker)
    except Exception as e:
        print(f"Error parsing ticker: {match}")
        print(f"Error message: {e}")

# Categorize tickers into US and non-US
us_tickers = []
non_us_tickers = []
for ticker in tickers_list:
    symbol = ticker['symbol']
    if '.' in symbol or symbol[0].isdigit():
        non_us_tickers.append(ticker)
    else:
        us_tickers.append(ticker)

# Prepare content for US_tickers.ts using JSON formatting
us_content = f"export const usTickers = {json.dumps(us_tickers)};\n"

# Prepare content for Non_US_tickers.ts using JSON formatting
non_us_content = f"export const nonUsTickers = {json.dumps(non_us_tickers)};\n"

# Write to US_tickers.ts
with open('US_tickers.ts', 'w', encoding='utf-8') as file:
    file.write(us_content)

# Write to Non_US_tickers.ts
with open('Non_US_tickers.ts', 'w', encoding='utf-8') as file:
    file.write(non_us_content)