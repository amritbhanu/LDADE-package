# Usage

We call the `LDADE` function by inputting a `UserConfig` class from `LDADE` module.

```python
class UserNullConfig(BaseConfig):
    """
    This is the inherited version of basic user config class
    ALL THE DATA INSIDE THIS CLASS IS NULL
    IT PROVIDES ALL THE INTERFACE VARIABLES
    """

    def __init__(self):
        BaseConfig.__init__(self)
        self.__dict__.update(
            F=None,
            CR=None,
            NP=None,
            GEN=None,
            Goal=None,
            data_samples=None,
            terms=None,
            fitness=None,
            max_iter=None,
            termination=None,
            random_state=None,
            learners_para=None,
            learners_para_bounds=None,
            learners_para_categories=None)


class UserTestConfig(BaseConfig):
    """
    This is the inherited version of basic user config class
    PRE-WRITTEN FOR THE USE OF TESTING CLASS
    CONTAINING ALL THE NECESSARY CONFIG INSIDE
    """

    def __init__(self):
        BaseConfig.__init__(self)
        self.__dict__.update(
            F=0.3,
            CR=0.7,
            NP=10,
            GEN=2,
            Goal="Max",
            data_samples=None,
            term=7,
            fitness=ldavem,
            max_iter=10,
            termination="Early",
            random_state=1,
            learners_para=OrderedDict([("n_components", 10),
                                       ("doc_topic_prior", 0.1),
                                       ("topic_word_prior", 0.01)]),
            learners_para_bounds=[(10, 100), (0.1, 1), (0.01, 1)],
            learners_para_categories=["integer", "continuous", "continuous"])
```

