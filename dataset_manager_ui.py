import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd

class DatasetManagerUI:
    def __init__(self, master):
        self.master = master
        master.title('Dataset Manager')

        self.upload_button = tk.Button(master, text='Upload CSV', command=self.upload_csv)
        self.upload_button.pack(pady=10)

        self.data_frame = ttk.Treeview(master, columns=('Column1', 'Column2', 'Column3'), show='headings')
        self.data_frame.heading('Column1', text='Column 1')
        self.data_frame.heading('Column2', text='Column 2')
        self.data_frame.heading('Column3', text='Column 3')
        self.data_frame.pack(pady=10)

        self.organize_button = tk.Button(master, text='Organize Datasets', command=self.organize_datasets)
        self.organize_button.pack(pady=10) 

        self.dataset_path = None

    def upload_csv(self):
        self.dataset_path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')])
        if self.dataset_path:
            self.show_data_preview()

    def show_data_preview(self):
        try:
            data = pd.read_csv(self.dataset_path)
            for i in self.data_frame.get_children():
                self.data_frame.delete(i)  # Clear previous data
            for index, row in data.iterrows():
                self.data_frame.insert('', 'end', values=row.tolist())
        except Exception as e:
            messagebox.showerror('Error', f'Could not read file: {e}')

    def organize_datasets(self):
        messagebox.showinfo('Information', 'Dataset organization feature not yet implemented.')

if __name__ == '__main__':
    root = tk.Tk()
    dataset_manager = DatasetManagerUI(root)
    root.mainloop()