"""
latent factor model to get item & user embeddings
"""
import tensorflow as tf

from common import NN


class Factor(NN):
    def __init__(
            self,
            n_users,
            n_items,
            model_spec,
            model_name=None,
    ):
        self.n_users, self.n_items = n_users, n_items
        self.model_spec = model_spec
        self.model_name = model_name if model_name is not None else "factor"

        super(Factor, self).__init__(graph=None)

    def initialization(
            self,
            item_emb_file=None,
            item_emb_data=None,
            user_emb_file=None,
            user_emb_data=None,
    ):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)
        self.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.v_emb_placeholder,
            fixed_input_init=self.v_emb_init,
            fixed_input_file=item_emb_file,
            fixed_input_data=item_emb_data,
        )
        self.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.u_emb_placeholder,
            fixed_input_init=self.u_emb_init,
            fixed_input_file=user_emb_file,
            fixed_input_data=user_emb_data,
        )

    def _setup_placeholder(self):
        with tf.variable_scope("placeholder"):
            self.u = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="users"
            )
            self.v = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="items"
            )
            # implicit #
            self.vn = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="items_neg"
            )

    def _setup_net(self):
        self._setup_emb()
        self.ue = ue = self._get_u_emb()
        ve = tf.nn.embedding_lookup(
            params=self.v_emb,
            ids=self.v
        )
        vne = tf.nn.embedding_lookup(
            params=self.v_emb,
            ids=self.vn
        )
        self.r_logit = self._setup_uv_interact(ue, ve)
        self.rn_logit = self._setup_uv_interact(ue, vne)

    def _setup_loss(self):
        # implicit bpr loss
        self.loss = tf.reduce_mean(1.0 - tf.sigmoid(self.r_logit - self.rn_logit))

    def _setup_optim(self):
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_spec['learning_rate']
        ).minimize(self.loss)

    def train(
            self,
            data_generator,
            epoch_start=0,
    ):
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.optimizer,
            op_losses=[self.loss],
            session=self.sess,
            batch_size=self.model_spec["batch_size"],
            max_epoch=self.model_spec["max_epoch"],
            epoch_start=epoch_start,
            op_savers=[self.saver],
            save_path_prefixs=[self.model_name],
            log_board_dir="../summary/" + self.model_name
        )
        return results

    def predict(
            self,
            data_generator
    ):
        results = self._feed_forward_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_predict,
            output=[self.r_logit, self.ue],
            session=self.sess,
            batch_size=self.model_spec["batch_size"]
        )
        return results

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
            self.u: data["u"],
            self.v: data["v"]
        }
        return feed_dict

    def _fn_feed_dict_train(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.u: data["u"][batch_index],
            self.v: data["v"][batch_index],
            self.vn: data["vn"][batch_index]
        }
        return feed_dict

    def _setup_uv_interact(
            self,
            ue,
            ve
    ):
        return tf.reduce_sum(ue * ve, axis=-1)

    def _get_u_emb(self):
        ue = tf.nn.embedding_lookup(
            params=self.u_emb,
            ids=self.u
        )
        return ue

    def _setup_emb(self):
        self.v_emb, self.v_emb_placeholder, self.v_emb_init = self.fixed_input_load(
            input_shape=[self.n_items, self.model_spec["emb_dim"]],
            trainable=True,
            name="v_emb"
        )
        self.u_emb, self.u_emb_placeholder, self.u_emb_init = self.fixed_input_load(
            input_shape=[self.n_users, self.model_spec["emb_dim"]],
            trainable=True,
            name="u_emb"
        )

    @staticmethod
    def _tf_emb(
            n,
            emb_dim,
            name=None
    ):
        emb = tf.Variable(
            initial_value=tf.random.normal(
                shape=[n, emb_dim],
                mean=0.0,
                stddev=0.01,
            ),
            dtype=tf.float32,
            name=name
        )
        return emb
