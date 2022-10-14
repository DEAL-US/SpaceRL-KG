from tkinter import *
from tkinter import ttk

def print_hierarchy(w, depth=0):
    print('  '*depth + w.winfo_class() + ' w=' + str(w.winfo_width()) + ' h=' + str(w.winfo_height()) + ' x=' + str(w.winfo_x()) + ' y=' + str(w.winfo_y()))
    for i in w.winfo_children():
        print_hierarchy(i, depth+1)

def calculate():
    print('placeholder')

root = Tk()
root.title("Model Generator")

# initial mainframe configuration.
# we make a 3 by 3 grid 
mainframe = ttk.Frame(root, padding="3 4 12 12")
mainframe.grid(column=0, row=0) #sticky=(N, W, E, S) <- makes the content stick to corner, we perfer it to be centered.
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# we create the input variable
feet = StringVar()
# and assign mainframe as its parent.
feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
# finally we assing its position in the grid (we start counting at 1 btw.)
feet_entry.grid(column=2, row=1, sticky=(W, E))

# same with the meters text.
meters = StringVar()
ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))

# the calculation button.
ttk.Button(mainframe, text="Calculate", command=calculate).grid(column=3, row=3, sticky=W)

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
root.bind("<Return>", calculate)

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