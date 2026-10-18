# Dataset snapshot

This folder contains the **Palmer Penguins** measurements: 344 adult penguins from
three species (Adelie, Chinstrap, Gentoo), observed on islands in the Palmer
Archipelago during 2007–2009. The CSV has eight columns. The model uses four numeric
measurements; `species` is its label. `island`, `sex`, and `year` remain available
for inspection but are not predictors.

The snapshot is bundled so a training run does not depend on GitHub availability.
`provenance.json` records its upstream commit, original URL, download date, and
SHA-256. The pipeline checks that fingerprint when it stages the bundled file.
Missing values in the upstream CSV are preserved. Do not fill them before splitting.

Source package: <https://allisonhorst.github.io/palmerpenguins/>

Credit: Horst AM, Hill AP, Gorman KB (2020), *palmerpenguins: Palmer Archipelago
(Antarctica) penguin data*, DOI: <https://doi.org/10.5281/zenodo.3960218>.
Measurements were collected and made available by Dr. Kristen Gorman and the
Palmer Station Long Term Ecological Research Program.

The upstream package declares **CC0**. See its
[DESCRIPTION](https://github.com/allisonhorst/palmerpenguins/blob/main/DESCRIPTION)
and the [CC0 dedication](https://creativecommons.org/publicdomain/zero/1.0/).
This attribution applies to the dataset, not the reference lab's source code.
