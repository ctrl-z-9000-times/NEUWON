from .control_panels import *
import numpy as np

class RegionEditor(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, "Region", init_settings_panel=False)
        self.key_parameter = "region_type"

        shape_options = ['Rectangle', 'Sphere', 'Cylinder']
        csg_options   = ['Union', 'Intersection', 'Not']
        options_grid  = np.array([shape_options, csg_options]).T
        self.add_button_create(radio_options={self.key_parameter: options_grid})
        self.add_button_delete()
        self.add_button_rename()
        self.add_button_duplicate()

        self.set_settings_panel(CustomSettingsPanel(self.get_widget(), self.key_parameter))

        geometry_kwargs = {
                "valid_range": (-np.inf, np.inf),
                "units": 'μm',
        }

        rectangle = self.settings.add_custom_settings_panel("Rectangle")
        rectangle.add_section("Rectangle")
        rectangle.add_entry("lower_x", **geometry_kwargs)
        rectangle.add_entry("lower_y", **geometry_kwargs)
        rectangle.add_entry("lower_z", **geometry_kwargs)
        rectangle.add_entry("upper_x", **geometry_kwargs)
        rectangle.add_entry("upper_y", **geometry_kwargs)
        rectangle.add_entry("upper_z", **geometry_kwargs)

        sphere = self.settings.add_custom_settings_panel("Sphere")
        sphere.add_section("Sphere")
        sphere.add_entry("center_x", **geometry_kwargs)
        sphere.add_entry("center_y", **geometry_kwargs)
        sphere.add_entry("center_z", **geometry_kwargs)
        sphere.add_entry("radius",   **geometry_kwargs)

        cylinder = self.settings.add_custom_settings_panel("Cylinder")
        cylinder.add_section("Cylinder")
        cylinder.add_entry("end_point_x", **geometry_kwargs)
        cylinder.add_entry("end_point_y", **geometry_kwargs)
        cylinder.add_entry("end_point_z", **geometry_kwargs)
        cylinder.add_entry("other_end_point_x", **geometry_kwargs)
        cylinder.add_entry("other_end_point_y", **geometry_kwargs)
        cylinder.add_entry("other_end_point_z", **geometry_kwargs)
        cylinder.add_entry("radius", **geometry_kwargs)

        union = self.settings.add_custom_settings_panel("Union")
        union.add_section("Union")

        intersection = self.settings.add_custom_settings_panel("Intersection")
        intersection.add_section("Intersection")

        inverse = self.settings.add_custom_settings_panel("Not")
        inverse.add_section("Not")

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
                sim[name] = (key, )
            elif key == 'Intersection':
                sim[name] = (key, )
            elif key == 'Not':
                sim[name] = (key, )

if __name__ == "__main__":
    print("RegionEditor Test")
    root = tk.Tk()
    RegionEditor(root).get_widget().grid()
    root.mainloop()
