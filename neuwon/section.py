
class Section(DB_Object):
    """ """
    @staticmethod
    def _initialize_database(db):
        cls = db.add_class("Section")

        return cls.get_instance_type()

    def __init__(self, nseg):
        1/0


    def _model_create_segment(self, parents, coordinates, diameters,
                shape="cylinder", shells=0, maximum_segment_length=np.inf):
        1/0 # TODO! this code was yanked out of the model class.
        """
        Argument parents:
        Argument coordinates:
        Argument diameters:
        Argument shape: must be "cylinder".
        Argument shells: unimplemented.
        Argument maximum_segment_length

        Returns a list of Segments.
        """
        if not isinstance(parents, Iterable):
            parents     = [parents]
            coordinates = [coordinates]
        parents_clean = np.empty(len(parents), dtype=Pointer)
        for idx, p in enumerate(parents):
            if p is None:
                parents_clean[idx] = NULL
            elif isinstance(p, Segment):
                parents_clean[idx] = p.entity.index
            else:
                parents_clean[idx] = p
        parents = parents_clean
        if not isinstance(diameters, Iterable):
            diameters = np.full(len(parents), diameters, dtype=Real)
        if   shape == "cylinder":   shape = np.full(len(parents), 1, dtype=np.uint8)
        else: raise ValueError("Invalid argument 'shape'")
        shape[parents == NULL] = 0 # Shape of root is always sphere.
        # This method only deals with the "maximum_segment_length" argument and
        # delegates the remaining work to the method: "_create_segment_batch".
        maximum_segment_length = float(maximum_segment_length)
        assert(maximum_segment_length > 0)
        if maximum_segment_length == np.inf:
            tips = self._create_segment_batch(parents, coordinates, diameters,
                    shapes=shape, shells=shells,)
            return [Segment(self, x) for x in tips]
        # Accept defeat... Batching operations is too complicated.
        tips = []
        old_coords = self.db.access("membrane/coordinates").get()
        old_diams = self.db.access("membrane/diameters").get()
        for p, c, d in zip(parents, coordinates, diameters):
            if p == NULL:
                tips.append(self._create_segment_batch([p], [c], [d], shapes=shape, shells=shells,))
                continue
            length = np.linalg.norm(np.subtract(old_coords[p], c))
            divisions = np.maximum(1, int(np.ceil(length / maximum_segment_length)))
            x = np.linspace(0.0, 1.0, num=divisions + 1)[1:].reshape(-1, 1)
            _x = np.subtract(1.0, x)
            seg_coords = c * x + old_coords[p] * _x
            if shape == 1:
                seg_diams = np.full(divisions, d)
            elif shape == 2:
                seg_diams = d * x + old_diams[p] * _x
            cursor = p
            for i in range(divisions):
                cursor = self._create_segment_batch([cursor],
                    seg_coords[i], seg_diams[i], shapes=shape, shells=shells,)[0]
                tips.append(cursor)
        return [Segment(self, x) for x in tips]

    def _model_create_segment_batch(self, parents, coordinates, diameters, shapes, shells):
        1/0 # TODO! this code was yanked out of the model class.
        num_new_segs = len(parents)
        m_idx = self.db.create_entity("membrane", num_new_segs, return_entity=False)
        i_idx = self.db.create_entity("inside", num_new_segs * (shells + 1), return_entity=False)
        o_idx = self.db.create_entity("outside", num_new_segs, return_entity=False)
        access = self.db.access
        access("membrane/inside")[m_idx]      = i_idx[slice(None,None,shells + 1)]
        access("inside/membrane")[i_idx]      = cp.repeat(m_idx, shells + 1)
        access("membrane/outside")[m_idx]     = o_idx
        access("membrane/parents")[m_idx]     = parents
        access("membrane/coordinates")[m_idx] = coordinates
        access("membrane/diameters")[m_idx]   = diameters
        access("membrane/shapes")[m_idx]      = shapes
        access("membrane/shells")[m_idx]      = shells
        children = access("membrane/children")
        write_rows = []
        write_cols = []
        for p, c in zip(parents, m_idx):
            if p != NULL:
                siblings = list(children[p].indices)
                siblings.append(c)
                write_rows.append(p)
                write_cols.append(siblings)
        data = [np.full(len(x), True) for x in write_cols]
        access("membrane/children", sparse_matrix_write=(write_rows, write_cols, data))
        primary    = access("membrane/primary")
        all_parent = access("membrane/parents").get()
        children   = access("membrane/children")
        for p, m in zip(parents, m_idx):
            if p == NULL: # Root.
                primary[m] = False # Shape of root is always sphere, value does not matter.
            elif all_parent[p] == NULL: # Parent is root.
                primary[m] = False # Spheres have no primary branches off of a them.
            else:
                # Set the first child added to a segment as the primary extension,
                # and all subsequent children as secondary branches.
                primary[m] = (children[p].getnnz() == 1)
        self._initialize_membrane_geometry(m_idx)
        self._initialize_membrane_geometry([p for p in parents if p != NULL])

        # shell_radius = [1.0] # TODO
        # access("inside/shell_radius")[i_idx] = cp.tile(shell_radius, m_idx)
        # 

        # TODO: Consider moving the extracellular tracking point to over the
        # center of the cylinder, instead of the tip. Using the tip only really
        # makes sense for synapses. Also sphere are special case.
        access("outside/coordinates")[m_idx] = coordinates
        self._initialize_outside(o_idx)
        return m_idx
