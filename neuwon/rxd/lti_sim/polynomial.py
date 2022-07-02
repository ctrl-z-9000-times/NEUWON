import numpy as np

class PolynomialForm:
    def __init__(self, inputs, polynomial):
        self.inputs = tuple(getattr(inp, 'name', inp) for inp in inputs)
        if isinstance(polynomial, PolynomialForm):
            self.terms = polynomial.terms
        elif len(self.inputs) == 1 and isinstance(polynomial, int):
            self.terms = tuple((power,) for power in range(polynomial + 1))
        else:
            self.terms = tuple(tuple(int(power) for power in term) for term in polynomial)
        self.num_terms = len(self.terms)
        assert self.num_terms > 0
        self.num_var_terms  = sum(any(power > 0 for power in term) for term in self.terms)
        self.degree         = max(sum(term) for term in self.terms)
        assert all(len(term) == len(self.inputs) for term in self.terms)
        assert all(all(power >= 0 for power in term) for term in self.terms)
        assert len(set(self.terms)) == self.num_terms, 'duplicate terms in polynomial detected'
        # Transform all polynomials into a canonical form.
        self.terms = tuple(sorted(self.terms, key=lambda term: (sum(term), tuple(reversed(term)))))

    def __len__(self):
        return self.num_terms

    def __str__(self):
        terms_list = []
        for term in self.terms:
            parts = []
            for inp, power in zip(self.inputs, term):
                if power > 1:
                    parts.append(f"{inp}^{power}")
                elif power == 1:
                    parts.append(f"{inp}")
                elif power == 0:
                    pass
            if len(parts) > 1:
                terms_list.append('(' + "*".join(parts) + ')')
            elif len(parts) == 1:
                terms_list.append(parts[0])
            elif len(parts) == 0:
                terms_list.append("1")
        return " + ".join(terms_list)

    def __eq__(self, other):
        return ((type(self) is type(other)) and
                (self.inputs == other.inputs) and
                (self.terms  == other.terms))

    def __hash__(self):
        return hash(self.inputs) ^ hash(self.terms)

    def _get_perimeter(self):
        num_inputs = len(self.inputs)
        terms_set  = set(self.terms)
        perimeter  = []
        for term in self.terms:
            for dim in range(num_inputs):
                next_term = tuple(power + int(idx == dim) for idx, power in enumerate(term))
                if next_term not in terms_set:
                    perimeter.append(term)
                    break
        return perimeter

    @classmethod
    def _from_perimeter(cls, inputs, perimeter) -> 'PolynomialForm':
        terms = set()
        for extent in perimeter:
            for inner in np.ndindex(*(x + 1 for x in extent)):
                terms.add(inner)
        return cls(inputs, terms)

    def _suggest(self, add_not_remove):
        num_inputs  = len(self.inputs)
        suggestions = set()
        for extent in self._get_perimeter():
            for dim in range(num_inputs):
                if add_not_remove:
                    new_extent = tuple(power + int(idx == dim) for idx, power in enumerate(extent))
                else:
                    new_extent = tuple(power - int(idx == dim) for idx, power in enumerate(extent))
                    if any(power < 0 for power in new_extent):
                        continue
                terms_set = set(self.terms)
                terms_set.remove(extent)
                terms_set.add(new_extent)
                new_poly = self._from_perimeter(self.inputs, terms_set)
                suggestions.add(new_poly)
        suggestions.discard(self)
        # Sort the suggestions from worst to best, so that the caller can pop
        # suggestions off of the end of the list.
        #   * Suggest removing the highest degree terms first.
        #   * Suggest adding the lowest degree terms first.
        return sorted(suggestions, key=lambda p: sum(sum(term) for term in p.terms), reverse=True)

    def suggest_add(self) -> ['PolynomialForm']:
        return self._suggest(True)

    def suggest_remove(self) -> ['PolynomialForm']:
        return self._suggest(False)
