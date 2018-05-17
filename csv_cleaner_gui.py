#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import signal
from tkinter import filedialog
from tkinter import *
 
#This is where we lauch the file manager bar.
def open_file(file_label):
    global FILE_NAME
    global root
    
    root.filename =  filedialog.askopenfilename(initialdir = os.getcwd(), 
                                                parent=root,
                                                title = "Select file", 
                                                filetypes = (("csv files","*.csv"),("all files","*.*")))
     
    if os.path.isfile(root.filename):
        file_label.config(text=root.filename, fg='black')
        file_label.pack(side=LEFT, pady=10, padx=10, anchor='n')
        FILE_NAME=root.filename
    else:
        file_label.config(text='An error occured!', fg='red')
    
    #Using try in case user types in unknown file or closes without choosing a file.
    #try:
    #    with open(root.filename ,'r') as f:
    #        print(f.read())
    #except:
    #    print("No file exists")

def close_app():
    global root
    root.destroy()
    
def clean_csv(file_label):
    file_label.config(text='Cleaning file...', fg='green')
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                           START SCRIPT
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

FILE_NAME = None

'''
# Main application prompt
'''
root = Tk()
def main():
    global root
    #root = Tk()
    root.resizable(width=False, height=False)
    root.geometry('{}x{}'.format(600, 200))
    
    
    title = root.title('CSV Prettify')
       
    # Add a button that allows the user to open the filedialog
    g = Frame(root)
    dirty_file = Button(g, 
                        text='Browse', 
                        width=12, 
                        command=lambda:open_file(file_label)).pack(
                                side=LEFT, pady=10, padx=10, anchor='n')

    
    # Add a label to display the status of the file
    file_label = Label(g, fg='red', text='No file selected')
    file_label.pack(side=LEFT, pady=10, padx=10, anchor='n')
    g.pack(anchor='nw')
    
    f = Frame(root)
    clean_btn = Button(f, 
                        text='Clean', 
                        width=10, 
                        command=lambda:clean_csv(file_label)).pack(
                                side=RIGHT, pady=10, padx=10, anchor='s')
    

                        
    # Add a button that allows the user to open the filedialog
    
    exit_btn = Button(f, 
                      text='Exit', 
                      width=10, 
                      command=lambda:close_app()).pack(
                              side=RIGHT, pady=10, padx=10, anchor='s')
    f.pack(side=BOTTOM, anchor='se')
    root.mainloop()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
