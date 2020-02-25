import numpy as np
import warnings
import math


class DataLoader(object):
    """
        random hold-out implicit prediction training
    """
    def __init__(
            self,
            data_file,
            n_items=None,
            n_users=None,
    ):
        self.n_items = n_items
        self.n_users = n_users
        self.data_raw = self.load_data(data_file=data_file)
        self.data_size = len(self.data_raw)

    def load_data(
            self,
            data_file
    ):
        """
        :param data_file: same as .interactor.Interactor.save_data()
        :return: implicit rating as np.array([[uid, iid]])
        """
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                uv = list(map(int, line.rstrip().split("\t")))
                if len(uv) > self.n_users:
                    print("data users larger than n_users")
                    raise ValueError
                data += [
                    [uid, uv[uid]] for uid in range(len(uv))
                ]
        data = np.array(data, dtype=np.int64)
        if np.any(data >= self.n_items):
            print("data item index larger than n_items")
            raise ValueError
        return data

    def generate(
            self,
            batch_size=1,
            random_shuffle=True
    ):
        index = np.arange(self.data_size)
        if random_shuffle:
            np.random.shuffle(index)
        if batch_size > self.data_size / 2:
            warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
                          (batch_size, self.data_size))
            batch_size = self.data_size
        max_batch = int(math.ceil(float(self.data_size) / float(batch_size)))
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(self.data_size, (i_batch + 1) * batch_size)]
            data_batch = self._process(
                data=self.data_raw[batch_index]
            )
            yield data_batch

    def _process(
            self,
            data,
            negative_sample="uniform"
    ):
        """
        :param data: part of self.data_raw
        :param negative_sample: method for negative sampling
        :return: {
            "u": np.array,
            "v": np.array,
            "vn": np.array
        }
        """
        data_out = {
            "u": data[:, 0],
            "v": data[:, 1],
        }
        if negative_sample == "uniform":
            data_out["vn"] = np.random.choice(
                self.n_items,
                size=data.shape[0]
            )
        else:
            raise NotImplementedError
        return data_out


class DataLoaderValid(DataLoader):
    """
        random hold-out implicit prediction valid
    """
    def __init__(
            self,
            data_file,
            n_items=None,
            n_users=None,
    ):
        self.n_items = n_items
        self.n_users = n_users
        self.data_raw = self.load_data(data_file=data_file)
        self.data_size = len(self.data_raw)
        self.data_u_dict = self.user_grouping(self.data_raw)
        self.n_users_effect = len(self.data_u_dict)
        self.i_user_effect = -1                  # count the current user index in n_users_effect

    def generate(
            self,
            batch_size=1,
            random_shuffle=False
    ):
        """
        for self.i_user_effect user in self.data_u_dict, provide candidate items (current: all)
        """
        self.i_user_effect += 1

        items = self.candidate_items()
        data_size = items.shape[0]
        index = np.arange(data_size)
        if random_shuffle:
            np.random.shuffle(index)
        if batch_size > data_size / 2:
            warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
                          (batch_size, data_size))
            batch_size = data_size
        max_batch = int(math.ceil(float(data_size) / float(batch_size)))
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(data_size, (i_batch + 1) * batch_size)]
            yield {
                "u": np.array(
                    [
                        self.data_u_dict[self.i_user_effect][0]
                    ],
                    dtype=np.int64
                ),
                "v": items[batch_index].astype(np.int64)
            }

    def candidate_items(self, **kwargs):
        return np.arange(self.n_items)

    @staticmethod
    def user_grouping(data):
        u_dict = {}
        for i in range(data.shape[0]):
            uid, iid = data[i][0], data[i][1]
            u_dict.setdefault(uid, []).append(iid)
        for uid in u_dict:
            u_dict[uid] = np.array(u_dict[uid], dtype=np.int64)
        u_dict_ordered = sorted(u_dict.items(), key=lambda x: x[0])
        return u_dict_ordered


class DataLoaderRecUsers(DataLoaderValid):
    """
        generate input for recommenders given users
    """
    def __init__(
            self,
            n_items,
            users
    ):
        self.n_items = n_items
        self.data_u_dict = list(map(
            lambda x: (x, None),
            users.tolist()
        ))
        self.i_user_effect = -1
        self.n_users_effect = len(self.data_u_dict)


class DataLoaderRecUsersCand(DataLoaderRecUsers):
    def __init__(
            self,
            users,
            candidates
    ):
        assert users.shape[0] == len(candidates)
        self.data_u_dict = [
            [users[i], candidates[i]] for i in range(users.shape[0])
        ]
        self.i_user_effect = -1
        self.n_users_effect = len(self.data_u_dict)

    def candidate_items(self, **kwargs):
        return self.data_u_dict[self.i_user_effect][1]


class DataLoaderItem(DataLoaderValid):
    """
        item-based data loader
    """
    def __init__(
            self,
            data_file,
            n_items=None,
            n_users=None,
            model_spec=None,
    ):
        self.model_spec = model_spec
        super(DataLoaderItem, self).__init__(
            data_file=data_file,
            n_items=n_items,
            n_users=n_users,
        )
        self.sequence_data = self.sequence_generate()
        self.data_size = len(self.sequence_data)

    def generate(
            self,
            batch_size=1,
            random_shuffle=False
    ):
        data_size = self.data_size
        index = np.arange(data_size)
        if random_shuffle:
            np.random.shuffle(index)
        if batch_size > data_size / 2:
            warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
                          (batch_size, data_size))
            batch_size = data_size
        max_batch = int(math.ceil(float(data_size) / float(batch_size)))
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(data_size, (i_batch + 1) * batch_size)]
            yield self.batch_process(batch_index)

    def batch_process(
            self,
            batch_index
    ):
        data = {
            "hist_v": [],
            "hist_length": [],
            "v": [],
            "vn": []
        }
        for i in batch_index.tolist():
            effect_length, input_seq, target = self.sequence_data[i]
            negative_sample = self.negative_sampling(self.n_items)
            data["hist_v"].append(input_seq)
            data["hist_length"].append(effect_length)
            data["v"].append(target)
            data["vn"].append(negative_sample)
        for k in data:
            if k == "hist_length":
                data[k] = np.array(data[k], dtype=np.float32)
            else:
                data[k] = np.array(data[k], dtype=np.int64)
        return data

    @staticmethod
    def negative_sampling(
            n_items,
            n_sample=1
    ):
        return np.random.choice(n_items, size=n_sample).squeeze()

    def sequence_generate(self):
        sequence_data = []
        for length in range(
            self.model_spec["min_length_sample"],
            self.model_spec["max_length_sample"] + 1
        ):
            sequence_data += self._sequence_generate(
                data_u_dict=self.data_u_dict,
                length=length,
                n=self.model_spec["n_seq_per_user"]
            )
            print("n_sequence=%d from length=%d" % (len(sequence_data), length))
        sequence_data = list(map(
            lambda x: [
                x[0].shape[0],
                self.pad_trunc(
                    seq=x[0],
                    length=self.model_spec["max_length_input"],
                    pad_index=self.n_items
                ),
                x[1]
            ],
            sequence_data
        ))
        return sequence_data

    @staticmethod
    def pad_trunc(
            seq,
            length,
            pad_index
    ):
        seq_length = seq.shape[0]
        if seq_length < length:
            seq_new = np.concatenate([
                seq,
                np.ones(length - seq_length, dtype=np.int64) * pad_index
            ])
        else:
            seq_new = seq[-length:]
        return seq_new

    @staticmethod
    def _sequence_generate(
            data_u_dict,
            length,
            n=1
    ):
        """
        from user data sample sequence input and target
        :param data_u_dict:
        :param length: input length
        :param n: number of samples per user
        :return: [[sequence, target]]
        """
        data = []
        for uid in range(len(data_u_dict)):
            seq = np.array(data_u_dict[uid][1])
            seq_length = seq.shape[0]
            if seq_length < length + n:
                continue                    # not enough sequence to build sample
            for i in range(n):
                sub_seq = np.random.choice(
                    seq,
                    size=(length + 1),
                    replace=False
                )
                data.append(
                    [sub_seq[:-1], sub_seq[-1]]
                )
        return data


class DataLoaderItemValid(DataLoaderItem):
    def __init__(
            self,
            data_file,
            n_items=None,
            n_users=None,
            model_spec=None,
    ):
        super(DataLoaderItemValid, self).__init__(
            data_file=data_file,
            n_items=n_items,
            n_users=n_users,
            model_spec=model_spec
        )
        self.i_sequence_effect = -1
        self.n_sequences_effect = self.data_size

    def generate(
            self,
            batch_size=1,
            random_shuffle=False
    ):
        self.i_sequence_effect += 1

        items = self.candidate_items()
        data_size = items.shape[0]
        index = np.arange(data_size)
        if random_shuffle:
            np.random.shuffle(index)
        if batch_size > data_size / 2:
            warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
                          (batch_size, data_size))
            batch_size = data_size
        max_batch = int(math.ceil(float(data_size) / float(batch_size)))
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(data_size, (i_batch + 1) * batch_size)]
            yield {
                "hist_v": np.array([self.sequence_data[self.i_sequence_effect][1]], dtype=np.int64),
                "hist_length": np.array([self.sequence_data[self.i_sequence_effect][0]], dtype=np.float32),
                "v": items[batch_index].astype(np.int64)
            }

    def candidate_items(
            self,
            **kwargs
    ):
        if "n_neg_sample" not in self.model_spec:
            return np.arange(self.n_items)
        else:
            cands = self.negative_sampling(
                n_items=self.n_items,
                n_sample=self.model_spec["n_neg_sample"]
            )
            return np.append(
                [self.sequence_data[self.i_sequence_effect][2]],
                cands,
                axis=0
            )


class DataLoaderItemPredict(DataLoaderItemValid):
    def __init__(
            self,
            n_items,
            hist_vs,
            max_length_input,
    ):
        self.n_items = n_items
        self.data_u_dict = hist_vs
        self.model_spec = {
            "max_length_input": max_length_input
        }
        self.sequence_data = self.sequence_generate()
        self.data_size = len(self.sequence_data)
        self.i_sequence_effect = -1
        self.n_sequences_effect = self.data_size

    def sequence_generate(self):
        sequence_data = self.data_u_dict
        sequence_data = list(map(
            lambda x: [
                x.shape[0],
                self.pad_trunc(
                    seq=x,
                    length=self.model_spec["max_length_input"],
                    pad_index=self.n_items
                ),
                None
            ],
            sequence_data
        ))
        return sequence_data
