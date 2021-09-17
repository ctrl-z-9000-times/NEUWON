import sympy
import os.path
import re

one = sympy.S.One

class Units:
    def __init__(self):
        self.library_path = os.path.join(os.path.split(__file__)[0], "nrnunits.lib.in")
        self.library_path = "nrnunits.lib.in"
        with open(self.library_path, 'rt') as f: library = f.read()
        library = [re.sub(r"^/.*", "", line.strip()) for line in library.split("\n")]
        library = [line.split(maxsplit=1) for line in library if line]
        library = [pair for pair in library if not pair[0].startswith("@LegacyY@")]
        self.prefixes = {}
        self.primitives = {}
        self.units = {}
        for name, value in library:
            if name.endswith("-"):
                self.prefixes[name[:-1]] = sympy.Rational(value)
            elif re.match(r"^\*.*\*$", value):
                enumeration = value[1:-1] # The library file assigns a letter (a-z) to each primative.
                self.primitives[name] = sympy.Symbol(name)
            else:
                if name.startswith("@LegacyN@"): name = name[len("@LegacyN@"):]
                self.add_unit(name, value)

    def standardize(self, units):
        """ Convert from the user specified units into NEUWON's prefixless unit system. """
        if units is None: return (one, one)
        factor, dimensions = self._parse_units(units)
        # TODO: Convert into NEUWON target units, which are not necessarily these units!
        return (factor, dimensions)

    def convert_to(self, value, dimensions, target_units):
        1/0

    def add_unit(self, name, string):
        if string in self.units:
            self.units[name] = self.units[string]
            return
        self.value_denom = False
        self.unit_denom = False
        try: prefix, dimensions = self._parse_units(string)
        except ValueError:
            print("Error while parsing units alias: \"%s = %s\"."%(name, string))
            raise
        self.units[name] = (prefix, dimensions.cancel())

    def __repr__(self):
        lines = []
        lines.append("Library path: " + self.library_path)
        lines.append("Primative Units:")
        for symbol in sorted((str(x) for x in self.primitives), key=lambda x: (x.upper(), x)):
            lines.append("    " + symbol)
        lines.append("Prefixes:")
        for name, value in sorted(self.prefixes.items(), key=lambda item: item[1]):
            lines.append("    " + name.ljust(10) + " = %g"%value)
        lines.append("Standard Units:")
        for name, (value, dimensions) in sorted(self.units.items(), key=lambda x: (x[0].upper(), x[0])):
            lines.append("    " + name.ljust(16) + " = %g, \t%s"%(value, dimensions))
        return "\n".join(lines)

    def _parse_units(self, string):
        """ Returns pair of (value, dimensions) where both are sympy expressions. """
        words = []
        for w in re.split(r"\s", string):
            if '|' in w:    words.extend(w.partition('|'))
            elif '/' in w:  words.extend(w.partition('/'))
            else:           words.append(w)
        for idx, w in enumerate(words): words[idx] = w.strip()
        words = [w for w in words if w]
        value = one; dimensions = one
        for idx, word in enumerate(words):
            if word == "-": pass
            elif word == "|": self.value_denom = True
            elif word == "/": self.unit_denom  = True
            else:
                word_value, word_dimensions = self._parse_word(word)
                if self.value_denom:
                    value = value / word_value
                    assert(word_dimensions == one)
                    self.value_denom = False
                if self.unit_denom:
                    value       = value / word_value
                    dimensions  = dimensions / word_dimensions
                else:
                    value       = value * word_value
                    dimensions  = dimensions * word_dimensions
        return (value, dimensions)

    def _parse_word(self, word):
        """ Returns pair of (value, dimensions) where both are sympy expressions. """
        # Check for plain old number. First check if python recognizes it as a
        # number, but prefer to use sympy.Rational for the actual arithmatic.
        try: float(word); return (sympy.Rational(word), one)
        except ValueError: pass
        # Check for scientific notation.
        if '-' in word or '+' in word:
            if   '-' in word: value, magnitude = word.split('-', maxsplit=1)
            elif '+' in word: value, magnitude = word.split('+', maxsplit=1)
            try: float(value); return (sympy.Rational(value) * 10 ** int(magnitude), one)
            except ValueError: pass
        # In all other contexts dashes are just word-separators.
        if '-' in word: return self._parse_units(word.replace('-', ' ', 1))
        # Someone forgot to put a space in between a number an the following word.
        missing_space = re.match(r"^(\d+)([a-zA-Z]+)$", word)
        if missing_space:
            number, word = missing_space.groups()
            return self._parse_units(number + " " + word)
        value = one;
        dimensions = one
        # Deal with any prefix.
        while True:
            for prefix in self.prefixes:
                if word.startswith(prefix):
                    value *= self.prefixes[prefix]
                    word = word[len(prefix):]
                    break
            else: break
        # Deal with any power.
        power = re.search(r"\d+$", word)
        if power is not None:
            power = int(power.group())
            word = re.sub(r"\d+$", "", word)
        else:
            power = 1
        # Lookup the name of the unit to find its actual value & dimensions.
        if word in self.primitives:
            dimensions = dimensions * self.primitives[word]
        elif word in self.units:
            value      = value      * self.units[word][0]
            dimensions = dimensions * self.units[word][1]
        elif not word: pass
        else: raise ValueError("Unrecognized unit \"%s\"."%word)
        return (value ** power, dimensions ** power)

builtin_units = Units()

if __name__ == "__main__":
    print("Built in units library")
    print(builtin_units)
