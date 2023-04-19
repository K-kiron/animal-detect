# LaTeX template for the Montreal AI Symposium

LaTeX template for the abstract submissions to the Montreal AI Symposium [(MAIS)](http://montrealaisymposium.com/)

## Provided files

* `mais.sty`: Style file
* `mais.bst`: Bibliography style file
* `instructions.tex` and `instructions.pdf`: documentation for the `mais` LaTeX package and tips for authors.
* `references.bib`: an example BibTeX file 
* `barebones_submission_template.tex`: a barebones submission template.
* `dummy_submission.tex` and `dummy_submission.pdf`: a dummy submission to easily see the approximate final appearance.

## Basic usage

The easiest way to start using this template is probably by cloning the repository:

```
git clone git@github.com:alexhernandezgarcia/mais-latex.git
```

Alternatively, you may simply download the files you will use. You may rename `barebones_submission_template.tex` and start writing your abstract there. If choose to use the default bibliography style and format, you can simply add the BibTex entries of your citations to `references.bib`.

Please read [`instructions.pdf`](./instructions.pdf) for a short introduction of the package and the main functionality.

## Questions

For questions and bug reports, please file issues at https://github.com/alexhernandezgarcia/mais-latex/issues

## Acknowledgements

This template heavily builds upon the template for [AutoML-Conf](https://www.automl.cc), developed by Roman Garnett, Frank Hutter and Marius Lindauer, and generously open-sourced in [https://github.com/automl-conf/LatexTemplate](https://github.com/automl-conf/LatexTemplate). The bibliography style file, `mais.bst`, has been slightly adapted from the style file for [ICML 2022](https://icml.cc/).
