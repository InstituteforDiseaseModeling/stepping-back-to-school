# Covasim controller

A set of functions and classes that create a feedback controller, based on an SEIR compartmental model, to specify the prevalence, incidence, or other *output* quantity of a Covasim simulation. The controller acts by performing closed-loop feedback on the output variable and one or more input parameters, most commonly beta.

Note: most of these functions and classes are not meant to be called directly; see the `school_tools` folder.