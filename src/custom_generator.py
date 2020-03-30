import keras


class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)

        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict

    def flow_from_dataframe(self,
                            dataframe,
                            directory=None,
                            x_col="filename",
                            y_col="class",
                            weight_col=None,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            validate_filenames=True,
                            **kwargs):

        for flowx, flowy in super().flow_from_dataframe(dataframe=dataframe,
                                                        directory=directory, x_col=x_col,
                                                        y_col=y_col,
                                                        batch_size=batch_size,
                                                        color_mode=color_mode,
                                                        seed=seed,
                                                        shuffle=shuffle,
                                                        class_mode=class_mode,
                                                        target_size=target_size):
            target_dict = {}
            i = 0
            for i, key in enumerate(kwargs['keys']):
                target_dict[key] = flowy[i]

            yield flowx, target_dict