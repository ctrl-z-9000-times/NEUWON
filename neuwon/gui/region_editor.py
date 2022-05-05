from .control_panels import *
import numpy as np
import sys

max_float = sys.float_info.max

class RegionEditor(ManagementPanel):
    def __init__(self, root):
        self.key_parameter = "region_type"
        super().__init__(root, "Region", custom_title=self.custom_title,
                controlled_panel=("CustomSettingsPanel", (self.key_parameter,)))

        shape_options = ['Rectangle', 'Sphere', 'Cylinder']
        csg_options   = ['Union', 'Intersection', 'Not']
        options_grid  = np.array([shape_options, csg_options]).T
        self.add_button_create(radio_options={self.key_parameter: options_grid})
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        geometry_kwargs = {
                "valid_range": (-max_float, max_float),
                "units": 'μm',
        }

        rectangle = self.controlled.add_custom_settings_panel("Rectangle")
        rectangle.add_entry("lower_x", **geometry_kwargs)
        rectangle.add_entry("lower_y", **geometry_kwargs)
        rectangle.add_entry("lower_z", **geometry_kwargs)
        rectangle.add_entry("upper_x", **geometry_kwargs)
        rectangle.add_entry("upper_y", **geometry_kwargs)
        rectangle.add_entry("upper_z", **geometry_kwargs)

        sphere = self.controlled.add_custom_settings_panel("Sphere")
        sphere.add_entry("center_x", **geometry_kwargs)
        sphere.add_entry("center_y", **geometry_kwargs)
        sphere.add_entry("center_z", **geometry_kwargs)
        sphere.add_entry("radius",   **geometry_kwargs)

        cylinder = self.controlled.add_custom_settings_panel("Cylinder")
        cylinder.add_entry("end_point_x", **geometry_kwargs)
        cylinder.add_entry("end_point_y", **geometry_kwargs)
        cylinder.add_entry("end_point_z", **geometry_kwargs)
        cylinder.add_entry("other_end_point_x", **geometry_kwargs)
        cylinder.add_entry("other_end_point_y", **geometry_kwargs)
        cylinder.add_entry("other_end_point_z", **geometry_kwargs)
        cylinder.add_entry("radius", **geometry_kwargs)

        def get_region_options():
            options = self.selector.get_list()
            options.remove(self.selector.get())
            return options

        union = self.controlled.add_custom_settings_panel("Union")
        union.add_dropdown("region_1", get_region_options)
        union.add_dropdown("region_2", get_region_options)

        intersection = self.controlled.add_custom_settings_panel("Intersection")
        intersection.add_dropdown("region_1", get_region_options)
        intersection.add_dropdown("region_2", get_region_options)

        inverse = self.controlled.add_custom_settings_panel("Not")
        inverse.add_dropdown("region", get_region_options)

    def custom_title(self, item):
        return f"{self.parameters[item]['region_type']}: {item}"

    def export(self):
        sim = {}
        for name, gui in self.get_parameters():
            key = gui[self.key_parameter]
            if key == 'Rectangle':
                sim[name] = (key,
                            [gui["lower_x"], gui["lower_y"], gui["lower_z"]],
                            [gui["upper_x"], gui["upper_y"], gui["upper_z"]])
            elif key == 'Sphere':
                sim[name] = (key,
                            [gui["center_x"], gui["center_y"], gui["center_z"]],
                            gui["radius"])
            elif key == 'Cylinder':
                sim[name] = (key,
                            [gui["end_point_x"], gui["end_point_y"], gui["end_point_z"]],
                            [gui["other_end_point_x"], gui["other_end_point_y"], gui["other_end_point_z"]],
                            gui["radius"])
            elif key == 'Union':
                sim[name] = (key, gui["region_1"], gui["region_2"])
            elif key == 'Intersection':
                sim[name] = (key, gui["region_1"], gui["region_2"])
            elif key == 'Not':
                sim[name] = (key, gui["region"])
        return sim

if __name__ == "__main__":
    root = tk.Tk()
    root.title("RegionEditor Test")
    RegionEditor(root).get_widget().grid()
    root.mainloop()
