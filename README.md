# SequenceModels

For example, for supervised learning tasks, there is an input block and a target block and we want to learn to predict targets from inputs. Learning to predict a cat/dog label (Label(["cat", "dog"])) from 2D images (Image{2}()) is a supervised image classification task.

A block is not a piece of data itself. Instead it describes the meaning of a piece of data in a context. That a piece of data is a block can be checked using [checkblock](block, data). A piece of data for the Label block above needs to be one of the labels, so checkblock(Label(["cat", "dog"]), "cat") == true, but checkblock(Label(["cat", "dog"]), "cat") == false.

We can say that a data container is compatible with a learning task if every observation in it is a valid sample of the sample block of the learning task. The sample block for supervised tasks is sampleblock = (inputblock, targetblock) so sample = getobs(data, i) from a compatible data container implies that checkblock(sampleblock, sample). This also means that any data stored in blocks must not depend on individual samples; we can store the names of possible classes inside the Label block because they are the same across the whole dataset
