"""
sequence recommendation model
"""
import tensorflow as tf

from recommender.factor import Factor


class Sequence(Factor):
    def __init__(
            self,
            n_items,
            model_spec,
            model_name=None,
    ):
        self.n_items = n_items
        self.model_spec = model_spec
        self.model_name = model_name

        super(Factor, self).__init__(graph=None)

    def initialization(
            self,
            item_emb_file=None,
            item_emb_data=None,
    ):
        self.sess.self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)
        self.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.v_emb_placeholder,
            fixed_input_init=self.v_emb_init,
            fixed_input_file=item_emb_file,
            fixed_input_data=item_emb_data,
        )

    def _setup_placeholder(self):
        with tf.variable_scope("placeholder"):
            self.hist_v = tf.placeholder(
                dtype=tf.int64,
                shape=[None, self.model_spec["max_hist_length"]],
                name="hist_items"
            )
            self.hist_length = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name="hist_length"
            )     # the effective length of hist_v
            # implicit #
            self.v = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="items"
            )
            self.vn = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="items_neg"
            )

    def _fn_feed_dict_train(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.hist_v: data["hist_v"][batch_index],
            self.v: data["v"][batch_index],
            self.vn: data["vn"][batch_index]
        }
        return feed_dict

    def _fn_feed_dict_predict(
            self,
            data,
            batch_index
    ):
        """

        :param data:
        :param batch_index: is not used to allow inputs of different size
        :return:
        """
        feed_dict = {
            self.hist_v: data["hist_v"],
            self.v: data["v"]
        }
        return feed_dict

    def _get_u_emb(self):
        hist_ve = tf.nn.embedding_lookup(
            params=self.v_emb,
            ids=self.hist_v
        )
        ue = self._setup_hist_aggregation(
            hist_ve=hist_ve,
            length=self.hist_length
        )
        return ue

    def _setup_hist_aggregation(
            self,
            hist_ve,
            length=None
    ):
        ue = tf.reduce_sum(hist_ve, axis=1) / tf.expand_dims(length, axis=-1)
        return ue

    def _setup_emb(self):
        self.v_emb, self.v_emb_placeholder, self.v_emb_init = self.fixed_input_load(
            input_shape=[self.n_items, self.model_spec["emb_dim"]],
            trainable=False,
            name="v_emb"
        )
