#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

def read_tsv(tsv_path: str,pd):
    """
    Reads a CSV file into a DataFrame.

    Args:
        tsv_path (str): The path to the CSV file to be read.
        pd: pandas as pd
    """

    try:

        # Open the CSV file in read mode and store the file object in the variable f.
        # The 'latin-1' encoding is used to ensure that the file is read correctly.
        with open(tsv_path, encoding='latin-1') as f:

            # Read the CSV file from the file object f into a DataFrame object named df.
            # The 'sep='\t' argument specifies that the separator is a tab character.
            df = pd.read_csv(f, sep='\t')

    except Exception as e:

        # Re-raise the exception with a more descriptive error message.
        # This will provide more context to the user about the error that occurred.
        raise Exception(f"Error reading CSV file: {tsv_path}") from e

    # Return the DataFrame object df.
    return df


    

def conversion_tsv_to_csv(tsv_path, csv_path, pd):
    """
    Convert a TSV file to a CSV file.

    Args:
        tsv_path (str): The path to the TSV file to convert.
        csv_path (str): The path to the CSV file to create.
        pd: pandas as pd.
    """

    # List of encodings to try
    encodings = ["latin-1", "utf-8", "utf-16"]

    # Try all encodings
    for encoding in encodings:
        try:
            # Load the TSV file into a DataFrame
            df = pd.read_csv(tsv_path, sep='\t', encoding=encoding)

            # Save the DataFrame to a CSV file
            df.to_csv(csv_path, index=False)

            # Print a success message
            print(f"The file was read successfully with the encoding: {encoding}")

            # Break out of the loop
            break
        except UnicodeDecodeError:
            # Print an error message
            print(f"Failed to read with the encoding: {encoding}")





# In[ ]:






# In[ ]:




