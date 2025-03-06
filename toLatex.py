from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

def ods_to_latex(file_path, sheet_index=0):
    doc = load(file_path)
    sheets = [table for table in doc.spreadsheet.getElementsByType(Table)]
    
    if sheet_index >= len(sheets):
        raise IndexError("Sheet index out of range.")
    
    sheet = sheets[sheet_index]
    
    data = []
    for row in sheet.getElementsByType(TableRow):
        row_data = []
        for col_index, cell in enumerate(row.getElementsByType(TableCell)):
            texts = [p.firstChild.data if p.firstChild else '' for p in cell.getElementsByType(P)]
            cell_value = ' '.join(texts).strip()  # Join multiple text elements and remove extra spaces
            cell_value = cell_value.replace("_", "\\_").replace("%", "\\%")  # Escape LaTeX characters
            
            # Ensure Timeout cells in columns B, C, D are explicitly captured
            if col_index in [1, 2, 3] and "Timeout" in cell_value:
                cell_value = "Timeout"
            
            # Ensure "-" is explicitly captured in columns E, F, G
            if col_index in [4, 5, 6] and "-" in cell_value:
                cell_value = "-"
            
            row_data.append(cell_value)
            
        data.append(row_data)
    
    num_cols = max(len(row) for row in data)
    latex_table = "\\begin{tabular}{" + "|c" * num_cols + "|} \\hline\n"
    
    for row in data:
        row += ["" ] * (num_cols - len(row))  # Ensure all rows have the same number of columns
        latex_table += ' & '.join(row) + " \\\\ \\hline\n"
    
    latex_table += "\\end{tabular}"
    
    return latex_table

# Example usage
if __name__ == "__main__":
    file_path = "4090_results_gecode.ods"  # Replace with your ODS file path
    latex_code = ods_to_latex(file_path)
    print(latex_code)
