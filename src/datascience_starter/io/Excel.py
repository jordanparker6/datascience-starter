import xlwings as xw
import pandas as pd
from typing import Union
from datascience_starter.base.logging import Logger

class Excel(Logger):
    """A class for interacting with Excel workbooks
    
    Args:
        file: The file name for the Excel workbook.

    """
    def __init__(self, file: str):
        super().__init__()
        self.file_name = file   #: The file name of the Excel workbook.
        self.wb = xw.Book(file) #: The xlwings workbook object.
        self.log.info(f"Excel File Read: {file}")

    def to_df(self, sheet: str, range: str = "A1", expand: str = 'table'):
        """Convert an Excel table to a pandas dataframe.

        Args:
            sheet: The sheet name of the workbook.
            range: The table starting range of the table.
            expand: The xlwing expand parameter (e.g. 'table' for table range or 'B10' for a range reference.

        Returns:
            A pandas dataframe.

        """
        df = pd.DataFrame(self.wb.sheets(sheet).range(range).expand(expand).value) 
        headers = df.iloc[0]
        df = pd.DataFrame(df.values[1:], columns=headers)
        return df

    def to_excel(self, df: pd.DataFrame, sheet: str, range: str = "A1", clear_range: Union[str, None] = None):
        """Output a pandas DataFrame to an excel range.

        Args:
            df: A pandas dataframe.
            sheet: The sheet name of the workbook.
            range: The starting range of the output.
            clear_rage: The range of the workbook to clear before printing (e.g. "A1:D10").
            
        """
        if clear_range is not None:
            self.wb.sheets(sheet).range(clear_range).clear_contents()
        print_range = wb.sheets(sheet).range(range) 
        print_range.value = df