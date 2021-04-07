import numpy as np
import os
import tempfile
import subprocess
from PIL import Image, ImageFont, ImageDraw

# Idea for API: "my_model.draw_image(pointer, f(v) -> [r,g,b], *args)"
# 
# The idea is to use the new Pointer API for specifying data channels to render.
# Also add an argument function to convert from value to the colorscale of their choice.

# TODO: Make these into methods on the model object, instead of global functions
# operating on the model. Then make these functions private.

def draw_image(model, segment_colors,
        output_filename, resolution,
        camera_coordinates,
        camera_look_at=(0,0,0),
        fog_color=(1,1,1),
        fog_distance=np.inf,
        lights=()):
    """ Use POVRAY to render an image of the model. """
    pov = ""
    pov += "camera { location <%s> look_at  <%s> }\n"%(
        ", ".join(str(x) for x in camera_coordinates),
        ", ".join(str(x) for x in camera_look_at))
    pov += "global_settings { ambient_light rgb<1, 1, 1> }\n"
    # TODO: Use the "lights" kw_arg to control these.
    pov += "light_source { <1, 0, 0> color rgb<1, 1, 1>}\n"
    pov += "light_source { <-1, 0, 0> color rgb<1, 1, 1>}\n"
    pov += "light_source { <0, 1, 0> color rgb<1, 1, 1>}\n"
    pov += "light_source { <0, -1, 0> color rgb<1, 1, 1>}\n"
    pov += "light_source { <0, 0, 1> color rgb<1, 1, 1>}\n"
    pov += "light_source { <0, 0, -1> color rgb<1, 1, 1>}\n"
    if fog_distance == np.inf:
        pov += "background { color rgb<%s> }\n"%", ".join(str(x) for x in fog_color)
    else:
        pov += "fog { distance %s color rgb<%s>}\n"%(str(fog_distance),
        ", ".join(str(x) for x in fog_color))
    for location in range(len(model)):
        parent = model.geometry.parents[location]
        coords = model.geometry.coordinates[location]
        # Special cases for root of tree, whos segment body is split between
        # it and its eldest child.
        if model.geometry.is_root(location):
            eldest = model.geometry.children[location][0]
            other_coords = (coords + model.geometry.coordinates[eldest]) / 2
        elif model.geometry.is_root(parent) and model.geometry.children[parent][0] == location:
            other_coords = (coords + model.geometry.coordinates[parent]) / 2
        else:
            other_coords = model.geometry.coordinates[parent]
        pov += "cylinder { <%s>, <%s>, %s "%(
            ", ".join(str(x) for x in coords),
            ", ".join(str(x) for x in other_coords),
            str(model.geometry.diameters[location] / 2))
        pov += "texture { pigment { rgb <%s> } } }\n"%", ".join(str(x) for x in segment_colors[location])
    pov_file = tempfile.NamedTemporaryFile(suffix=".pov", mode='w+t', delete=False)
    pov_file.write(pov)
    pov_file.close()
    subprocess.run(["povray",
        "-D", # Disables immediate graphical output, save to file instead.
        "+O" + output_filename,
        "+W" + str(resolution[0]),
        "+H" + str(resolution[1]),
        pov_file.name,],
        stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL,
        check=True,)
    os.remove(pov_file.name)

class Animation:
    def __init__(self, model,
            skip = 0,
            **kw_args):
        """
        Argument skip: Don't render this many frames between every actual render.
        """
        self.model = model
        self.kw_args = kw_args
        self.frames_dir = tempfile.TemporaryDirectory()
        self.frames = []
        self.skip = int(skip)
        self.ticks = 0

    def add_frame(self, colors,
            shrink = None,
            text = ""):
        """
        Argument shrink: scale the image dimensions by this number to reduce filesize.
        Argument text: is overlayed on the top right corner.
        """
        self.ticks += 1
        if self.ticks % (self.skip+1) != 0:
            return
        self.frames.append(os.path.join(self.frames_dir.name, str(len(self.frames))+".png"))
        draw_image(self.model, colors, self.frames[-1], **self.kw_args)
        if text or shrink:
            img = Image.open(self.frames[-1])
            if shrink is not None:
                new_size = (int(round(img.size[0] * shrink)), int(round(img.size[1] * shrink)))
                img = img.resize(new_size, resample=Image.LANCZOS)
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), text, (0, 0, 0))
            img.save(self.frames[-1])

    def save(self, output_filename):
        """ Save into a GIF file that loops forever. """
        self.frames = [Image.open(i) for i in self.frames] # Load all of the frames.
        dt = (self.skip+1) * self.model.time_step * 1e3
        self.frames[0].save(output_filename, format='GIF',
                append_images=self.frames[1:], save_all=True,
                duration=int(round(dt * 1e3)), # Milliseconds per frame.
                optimize=True, quality=0,
                loop=0,) # Loop forever.
