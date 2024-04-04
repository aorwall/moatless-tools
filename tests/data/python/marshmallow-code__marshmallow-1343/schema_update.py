class BaseSchema(base.SchemaABC):
    # ... other code

    def _invoke_field_validators(self, unmarshal, data, many):
        if data is None:
            return

        for attr_name in self.__processors__[(VALIDATES, False)]:
            validator = getattr(self, attr_name)
            validator_kwargs = validator.__marshmallow_kwargs__[(VALIDATES, False)]
            field_name = validator_kwargs['field_name']

            try:
                field_obj = self.fields[field_name]
            except KeyError:
                if field_name in self.declared_fields:
                    continue
                raise ValueError('"{0}" field does not exist.'.format(field_name))

            if many:
                for idx, item in enumerate(data):
                    if item is None:
                        continue
                    try:
                        value = item[field_obj.attribute or field_name]
                    except KeyError:
                        pass
                    else:
                        validated_value = unmarshal.call_and_store(
                            getter_func=validator,
                            data=value,
                            field_name=field_obj.load_from or field_name,
                            field_obj=field_obj,
                            index=(idx if self.opts.index_errors else None)
                        )
                        if validated_value is missing:
                            data[idx].pop(field_name, None)
            else:
                try:
                    value = data[field_obj.attribute or field_name]
                except KeyError:
                    pass
                else:
                    validated_value = unmarshal.call_and_store(
                        getter_func=validator,
                        data=value,
                        field_name=field_obj.load_from or field_name,
                        field_obj=field_obj
                    )
                    if validated_value is missing:
                        data.pop(field_name, None)
    # ... other code
