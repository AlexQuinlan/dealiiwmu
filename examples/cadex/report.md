

# CCX/ABQ Input Files for Deal.ii

## MATERIALS

### Naming the Material

Materials should be listed in the form 'Material-N', where 'N' is an
integer ID number for the material.  This should work to assign material
id N to the cells within deal.ii.

### Element sets for a material

ELSET=EB101 sets the material_id to 101. This is 
set from *Element, Type=S8, ELSET=EB101.



### Assigning the Material

The standard CCX keyword *SOLID SECTION is used to attach a material to a given
element set.


### Keyword differences

Element sets defined as part of the *Element keyword
behavior differently than those specified with the
*ELSET keyword.  Those specified with *Element have their
material-id set equal to the Elset number.  Those specified
using *ELSET have their elset number set equal to the
Material number.

### Using the material withing deal.ii

If the steps are followed,
a given cell will have an assigned material-id number.  This can be
referenced withing the C++ job file via
	   cell->material_id()
which can be used to determine if a given cell belongs to a certain set.


Additionally, the optional argument "apply_all_indicators_to_manifolds"=true can be used
when reading the file:

     gi.read_abaqus(in, true)

This will result in the manifold-id of each cell being set to its material-id.


## SURFACES

### Defining a surface
Element surfaces can be specified using the typical CCX approach,
though the naming must follow the convention of SSN, where N is an integer.

       *Surface, type=ELEMENT, name=SS101
       AQELSET01, S3 

This will results in the specified faces having a manifold_id equal to
101. The elements in the surface can either be specified individually
or by an elset.

### Defining an element set for a surface

Element sets can be named arbitrarily, i.e. they do not need to follow
the "EB101" convention used in the material elset definition.  All
ELSETs are added to a list of ELSETs within the abaqus reader.
However, this list is NOT returned from the reading process, and
therefore all information must be transferred through the material_id or
manifold_id.

The ELSET reader can recognize the 'Generate' keyword used in CCX/ABQ

Element sets do not append.  If we have elset definitions:
	*ELSET=EB11
	1,2,3
	*ELSET=EB11
	4,5,6
the second declaration replaces the first, rather than adding to it.

Reminder: Element numbers and cell numbers will differ by 1, since the cell count starts at 0
within C++.

### Faces

Hex cell faces are defined as:

1. F1: 1,4,3,2
2. F2: 5,8,7,6
3. F3: 1,2,6,5
4. F4: 2,3,7,6
5. F5: 3,4,8,7
6. F6: 1,5,8,4


### Nodes

Nodes I do not see any means of passing node sets or nodal surfaces
through the current deal.ii development. (In fact, the deal.ii
comments say that NSETs are ignored because they have no use for
them).  However, we may be able to add in our own routines to deal
with this.

[comment]: <> ## Additional examples on Material-id assignment
[comment]: <> Here's what I've gathered as the ID rules:
[comment]: <> -If an element is included in a set defined as part of an element definition, i.e.
[comment]: <>     *Element, Elset=EB55
[comment]: <> it will be given a Material-id equal to the elset id (55 in this case).
[comment]: <> -If that Elset is later assigned a material via section definition, i.e.
[comment]: <>     *Solid Section, Elset=EB55, Material=Material-77
[comment]: <> it will retain the elset number as the material-id
[comment]: <> -If an element within EB55 is added to another elset, i.e.
[comment]: <>     *Elset, elset=EB105
[comment]: <> it will retain the original elset number as the material-id (55 in this case)
[comment]: <> -However, if the new elset is used in a material definition, i.e.
[comment]: <>     *Solid Section, Elset=EB105, Material=Material-33
[comment]: <> the new elset's material-id will be assigned as the cell material-id (33 in this case)



[comment]: <> Removing the 'apply_all_indicators_to_manifolds=true' argument for
[comment]: <> read_abq results in the same material-id application, but the
[comment]: <> manifold-id being a random number