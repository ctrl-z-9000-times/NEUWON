from .control_panels import *
from neuwon.rxd.nmodl.parser import NmodlParser
from tkinter import filedialog
import os.path

class MechanismManager(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Mechanism", controlled_panel=("CustomSettingsPanel", ("filename",)))
        # 
        self.selector.add_button("Import", self.import_mechanisms)
        self.add_button_delete("Remove")
        self.add_button_rename(row=1)
        self.selector.add_button("Info", self.info_on_mechanism, require_selection=True, row=1)
        self.documentation = {}

    def import_mechanisms(self, selected):
        files = filedialog.askopenfilenames(
                title="Import Mechanisms",
                filetypes=[('NMODL', '.mod')])
        for abspath in files:
            name = os.path.splitext(os.path.basename(abspath))[0]
            if name in self.parameters:
                continue
            self.parameters[name] = {'filename': abspath}
            self._make_nmodl_settings_panel(abspath)
            self.selector.insert(name)

    def set_parameters(self, parameters):
        for mech_name, mech_parameters in parameters.items():
            filename = mech_parameters["filename"]
            try:
                self.controlled.get_panel(filename)
            except KeyError:
                self._make_nmodl_settings_panel(filename)
        super().set_parameters(parameters)

    def _make_nmodl_settings_panel(self, filename):
        settings_panel = self.controlled.add_settings_panel(filename, override_mode=True)
        parser = NmodlParser(filename, preprocess=False)
        for name, (value, units) in parser.gather_parameters().items():
            settings_panel.add_entry(name, title=name, default=value, units=units)
        name, point_process, title, description = parser.gather_documentation()
        self.documentation[filename] = title + "\n\n" + description

    def info_on_mechanism(self, selected):
        window, frame = Toplevel(selected + " Documentation")
        # Display filename in a raised box.
        filename = self.parameters[selected]["filename"]
        fn = ttk.Label(frame, text=filename, padding=padx, relief='raised')
        fn.grid(row=0, column=0, padx=padx, pady=pad_top, sticky='e')
        # Button to copy the filename to the clipboard.
        def copy_filename():
            window.clipboard_clear()
            window.clipboard_append(filename)
        copy = ttk.Button(frame, text="Copy", command=copy_filename)
        copy.grid(row=0, column=1, padx=padx, pady=pad_top, sticky='w')
        # Show documentation scraped from the NMODL file.
        docs = ttk.Label(frame, text=self.documentation[filename], justify='left', padding=padx)
        docs.grid(row=1, column=0, columnspan=2, padx=padx, pady=pady)

    @classmethod
    def export(cls, parameters):
        sim = {}
        for name, gui in parameters.items():
            gui = dict(gui)
            sim[name] = (gui.pop("filename"), gui)
        return sim

class MechanismSelector(ManagementPanel):
    def __init__(self, root, mechanism_manager):
        super().__init__(root, "Mechanism")
        self.mechanisms = mechanism_manager
        # 
        self.selector.add_button("Insert", self.insert_mechanism)
        self.add_button_delete("Remove", require_confirmation=False)
        self.selector.add_button("Info", self.mechanisms.info_on_mechanism, require_selection=True, row=1)
        # 
        self.controlled.add_empty_space()
        self.controlled.add_entry('magnitude', default=1.0)

    def insert_mechanism(self, selected):
        window, frame = Toplevel("Select Mechanisms to Insert")
        mechanisms = sorted(self.mechanisms.parameters)
        listbox = tk.Listbox(frame, selectmode='extended', exportselection=True)
        listbox.grid(row=0, column=0, columnspan=2, padx=padx, pady=pad_top)
        listbox.insert(0, *mechanisms)
        selection = []
        def ok_callback():
            for idx in listbox.curselection():
                selection.append(mechanisms[idx])
            window.destroy()
        ok = ttk.Button(frame, text="Ok",     command=ok_callback,)
        no = ttk.Button(frame, text="Cancel", command=window.destroy,)
        ok.grid(row=1, column=0, padx=2*padx, pady=pad_top)
        no.grid(row=1, column=1, padx=2*padx, pady=pad_top)
        # 
        listbox.focus_set()
        listbox.bind("<Double-Button-1>", lambda event: ok_callback())
        window .bind("<Escape>", lambda event: window.destroy())
        # Make the dialog window modal. This prevents user interaction with
        # any other application window until this dialog is resolved.
        window.grab_set()
        window.transient(self.frame)
        window.wait_window(window)
        # 
        for x in selection:
            if x in self.parameters:
                continue
            self.parameters[x] = {}
            self.selector.insert(x)

if __name__ == "__main__":
    root = tk.Tk()
    MechanismManager(root).get_widget().grid()
    root.mainloop()
