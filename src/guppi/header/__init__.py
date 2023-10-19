from rao_keyvalue_property_mixin_classes.guppi_raw import GuppiRawProperties
from rao_keyvalue_property_mixin_classes.hpdaq_ata import HpdaqAtaProperties
from rao_keyvalue_property_mixin_classes.hpdaq_cosmic import HpdaqCosmicProperties
from rao_keyvalue_property_mixin_classes.hpdaq_meerkat import HpdaqMeerkatProperties


class GuppiRawFitsExportable:
    @staticmethod
    def _keyvalue_to_fits(key: str, value) -> str:
        v = str(value) if not isinstance(value, str) else f"\'{value[:69]}\'"
        return f"{key[:8]:8s}={v[:71]:71s}"

    def to_fits(self) -> str:
        return "".join(
            GuppiRawFitsExportable._keyvalue_to_fits(key, value)
            for key, value in self.items()
        ) + "END                                                                             "


class GuppiRawHeader(dict, GuppiRawProperties, GuppiRawFitsExportable):
    pass


class GuppiRawAtaHeader(HpdaqAtaProperties, GuppiRawHeader):
    pass


class GuppiRawCosmicHeader(HpdaqCosmicProperties, GuppiRawHeader):
    pass


class GuppiRawMeerkatHeader(HpdaqMeerkatProperties, GuppiRawHeader):
    pass


PROPERTY_FGET_CLASS_MAP = {
    GuppiRawProperties.telescope.fget: {
        cls.telescope.fget(): cls
        for cls in [
            GuppiRawAtaHeader,
            GuppiRawCosmicHeader,
            GuppiRawMeerkatHeader
        ]
    }
}


def auto_init_GuppiRawHeader(keyvalues: dict):
    for property_fget, value_class_map in PROPERTY_FGET_CLASS_MAP.items():
        property_value = property_fget(keyvalues)
        if property_value in value_class_map:
            return value_class_map[property_value](**keyvalues)
    return GuppiRawHeader(**keyvalues)
