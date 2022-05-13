from .control_panels import *

class RegionEditor(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, 'Region', custom_title=self.custom_title,
                         panel=('CustomSettingsPanel', ('region_type',)))

        shape_options = ['Rectangle', 'Sphere', 'Cylinder']
        csg_options   = ['Union', 'Intersection', 'Not']
        options_grid  = np.array([shape_options, csg_options]).T
        self.add_button_create(radio_options={'region_type': options_grid})
        self.add_button_delete()
        self.add_button_rename(row=1)
        self.add_button_duplicate(row=1)

        geometry_kwargs = {
                'valid_range': (-max_float, max_float),
                'units': 'μm',
        }

        rectangle = self.controlled.add_settings_panel('Rectangle')
        rectangle.add_entry('lower_x', **geometry_kwargs)
        rectangle.add_entry('lower_y', **geometry_kwargs)
        rectangle.add_entry('lower_z', **geometry_kwargs)
        rectangle.add_entry('upper_x', **geometry_kwargs)
        rectangle.add_entry('upper_y', **geometry_kwargs)
        rectangle.add_entry('upper_z', **geometry_kwargs)

        sphere = self.controlled.add_settings_panel('Sphere')
        sphere.add_entry('center_x', **geometry_kwargs)
        sphere.add_entry('center_y', **geometry_kwargs)
        sphere.add_entry('center_z', **geometry_kwargs)
        sphere.add_entry('radius',   **geometry_kwargs)

        cylinder = self.controlled.add_settings_panel('Cylinder')
        cylinder.add_entry('end_point_x', **geometry_kwargs)
        cylinder.add_entry('end_point_y', **geometry_kwargs)
        cylinder.add_entry('end_point_z', **geometry_kwargs)
        cylinder.add_entry('other_end_point_x', **geometry_kwargs)
        cylinder.add_entry('other_end_point_y', **geometry_kwargs)
        cylinder.add_entry('other_end_point_z', **geometry_kwargs)
        cylinder.add_entry('radius', **geometry_kwargs)

        def get_region_options():
            options = self.selector.get_list()
            options.remove(self.selector.get())
            return options

        union = self.controlled.add_settings_panel('Union')
        union.add_dropdown('region_1', get_region_options)
        union.add_dropdown('region_2', get_region_options)

        intersection = self.controlled.add_settings_panel('Intersection')
        intersection.add_dropdown('region_1', get_region_options)
        intersection.add_dropdown('region_2', get_region_options)

        inverse = self.controlled.add_settings_panel('Not')
        inverse.add_dropdown('region', get_region_options)

    def custom_title(self, item):
        return f"{self.parameters[item]['region_type']}: {item}"

def export(parameters: dict) -> dict:
    for name, rgn in parameters.items():
        region_type = rgn['region_type']

        if region_type == 'Rectangle':
            parameters[name] = (region_type,
                        [rgn['lower_x'], rgn['lower_y'], rgn['lower_z']],
                        [rgn['upper_x'], rgn['upper_y'], rgn['upper_z']])

        elif region_type == 'Sphere':
            parameters[name] = (region_type,
                        [rgn['center_x'], rgn['center_y'], rgn['center_z']],
                        rgn['radius'])

        elif region_type == 'Cylinder':
            parameters[name] = (region_type,
                        [rgn['end_point_x'], rgn['end_point_y'], rgn['end_point_z']],
                        [rgn['other_end_point_x'], rgn['other_end_point_y'], rgn['other_end_point_z']],
                        rgn['radius'])

        elif region_type == 'Union':
            parameters[name] = (region_type, rgn['region_1'], rgn['region_2'])

        elif region_type == 'Intersection':
            parameters[name] = (region_type, rgn['region_1'], rgn['region_2'])

        elif region_type == 'Not':
            parameters[name] = (region_type, rgn['region'])

        else:
            raise NotImplementedError(region_type)
    return parameters
