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
mainframe = ttk.Frame(root, padding="4 4 12 12")
mainframe.grid(column=0, row=0) #sticky=(N, W, E, S) <- makes the content stick to corner, we perfer it to be centered.
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# STYLE FOR ELEMENTS.
s = ttk.Style()
s.configure('Danger.TFrame', background='red', borderwidth=5, relief='raised')
# ttk.Frame(root, width=200, height=200, style='Danger.TFrame').grid()

textstyle = ttk.Style()
textstyle.configure('general purpose text.')

# ENDSTYLE

feet = StringVar()
# as we want to reference this variable later we cannot add grid directly as it returs nothing.
# entry widget is a free text box.
feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
feet_entry.grid(column=2, row=1, sticky=(W, E))

# the ouput of the program goes here.
meters = StringVar()
ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))

# button that triggers the calculation.
ttk.Button(mainframe, text="Calculate", command=placeholder).grid(column=3, row=3, sticky=W)

# check button
measureSystem = StringVar()
check = ttk.Checkbutton(mainframe, text='Use Metric', command=placeholder, variable=measureSystem,onvalue='metric', offvalue='imperial')

# radio buttons
phone = StringVar()
home = ttk.Radiobutton(mainframe, text='Home', variable=phone, value='home')
office = ttk.Radiobutton(mainframe, text='Office', variable=phone, value='office')
cell = ttk.Radiobutton(mainframe, text='Mobile', variable=phone, value='cell')

# and some labels
ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)

# adding pixel padding to all children of mainframe.
for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)


#automatically focus on the feet entry.
feet_entry.focus()
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