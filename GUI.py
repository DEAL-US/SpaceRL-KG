from tkinter import *
from tkinter import ttk

def print_hierarchy(w, depth=0):
    print('  '*depth + w.winfo_class() + ' w=' + str(w.winfo_width()) + ' h=' + str(w.winfo_height()) + ' x=' + str(w.winfo_x()) + ' y=' + str(w.winfo_y()))
    for i in w.winfo_children():
        print_hierarchy(i, depth+1)

def placeholder():
    print('placeholder')

root = Tk()
root.title("Model Generator")

# Initial mainframe configuration.
# 3 columns and 4 rows.
mainframe = ttk.Frame(root, padding="12 12 12 12")
mainframe.grid(column=0, row=0)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# region STYLES
s = ttk.Style()
# s.configure('Danger.TFrame', background='red', borderwidth=5, relief='raised')
t = ttk.Style()
# endregion STYLES


# region ELEMENTS

# text entries
namelbl = ttk.Label(mainframe, text="Name")
name = StringVar()
name_entry = ttk.Entry(mainframe, textvariable=name, text="name")

surnamelbl = ttk.Label(mainframe, text="Surname")
surname = StringVar()
surname_entry = ttk.Entry(mainframe, textvariable=surname, text="surname")

# radio buttons
genderlbl = ttk.Label(mainframe, text='Gender')
gender = StringVar()
male = ttk.Radiobutton(mainframe, text='Woman', variable=gender, value='F') #Female
female = ttk.Radiobutton(mainframe, text='Man', variable=gender, value='M') #Male
other = ttk.Radiobutton(mainframe, text='Other', variable=gender, value='O') #Other

# check buttons
newscheck = BooleanVar(value=False)
check = ttk.Checkbutton(mainframe, text='get newsletter?', variable=newscheck)
test1 = BooleanVar(value=False)
check2 = ttk.Checkbutton(mainframe, text='test1', variable=test1)
test2 = BooleanVar(value=False)
check3 = ttk.Checkbutton(mainframe, text='test2', variable=test2)
test3 = BooleanVar(value=False)
check4 = ttk.Checkbutton(mainframe, text='test3', variable=test3)

# Functional Buttons
config_button = ttk.Button(mainframe, text="Configuration", command=placeholder)


# endregion ELEMENTS

# GRID MANAGEMENT
# Sticky options can choose where in the cell the element is 

#row0
namelbl.grid(column=0, row=0, sticky=['E'])
name_entry.grid(column=1, row=0, columnspan=2)
surnamelbl.grid(column=3, row=0, sticky=['E'])
surname_entry.grid(column=4, row=0, columnspan=2)

#row1
genderlbl.grid(row=1, column=0, sticky=['E'])
male.grid(row=1, column=1)
female.grid(row=1, column=2)
other.grid(row=1, column=3)

config_button.grid(row=1, column=4, columnspan=2, rowspan=2)

#row2
check.grid(row=2, column=0)
check2.grid(row=2, column=1)
check3.grid(row=2, column=2)
check4.grid(row=2, column=3)

# END GRID MANAGEMENT

# adding pixel padding to all children of mainframe.
for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)


#automatically focus on the feet entry.
name_entry.focus()
# if <Return> key is pressed we run the calculate function.
root.bind("<Return>", placeholder)

# some more binding to keys.
# l = ttk.Label(mainframe, text="Starting...")
# l.grid(column=2, row=4, sticky=S)

# l.bind('<Enter>', lambda e: l.configure(text='Moved mouse inside'))
# l.bind('<Leave>', lambda e: l.configure(text='Moved mouse outside'))
# l.bind('<ButtonPress-1>', lambda e: l.configure(text='Clicked left mouse button'))
# l.bind('<3>', lambda e: l.configure(text='Clicked right mouse button'))
# l.bind('<Double-1>', lambda e: l.configure(text='Double clicked'))
# l.bind('<B3-Motion>', lambda e: l.configure(text='right button drag to %d,%d' % (e.x, e.y)))

# a complete list of bindings is avaliable at https://tkdocs.com/tutorial/concepts.html

print_hierarchy(root)

root.mainloop()