# noveList
My project for [Insight Boston](http://www.insightboston.org/) Health Data Science 2020A. Predicts book ratings based on data from [Goodreads](https://www.goodreads.com/), collected from the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home?authuser=0). All code associated with creating the model can be found at my [insight](https://github.com/megthommes/insight) repository.

# Motivation
[Goodreads](https://www.goodreads.com/) is a "social cataloging website" that enables users to track which books they've read and would like to read, review the books they have read, and much more. However, there is no easy way to sort through your "To-Read" books. [NoveList](https://insight-novelist.herokuapp.com/) takes books you have read and books you would like to read, and predicts how much you would enjoy the books you want to read (from 1 to 5 stars).

# Built With
- Python
- [streamlit](https://www.streamlit.io/)
- [Heroku](https://www.heroku.com)

Dependencies can be found in the environment.yml file, and this file can be used to create a conda environment with
```console
foo@bar:~$ conda env create -f environment.yml
```

# License
MIT @ [Meghan Thommes](meghanthommes.com)
