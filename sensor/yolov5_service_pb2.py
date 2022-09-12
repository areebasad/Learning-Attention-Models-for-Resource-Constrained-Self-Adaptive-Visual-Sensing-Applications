# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yolov5_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='yolov5_service.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x14yolov5_service.proto\"\x15\n\x05Image\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"3\n\x0f\x44\x65tectedObjects\x12 \n\x07objects\x18\x01 \x03(\x0b\x32\x0f.DetectedObject\"m\n\x0e\x44\x65tectedObject\x12\x12\n\nclass_name\x18\x01 \x01(\t\x12\x11\n\tclass_idx\x18\x02 \x01(\r\x12\x12\n\x02p1\x18\x03 \x01(\x0b\x32\x06.Point\x12\x12\n\x02p2\x18\x04 \x01(\x0b\x32\x06.Point\x12\x0c\n\x04\x63onf\x18\x05 \x01(\x01\"\x1d\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x32,\n\x06YoloV5\x12\"\n\x06\x64\x65tect\x12\x06.Image\x1a\x10.DetectedObjectsb\x06proto3')
)




_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='Image.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=45,
)


_DETECTEDOBJECTS = _descriptor.Descriptor(
  name='DetectedObjects',
  full_name='DetectedObjects',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='objects', full_name='DetectedObjects.objects', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=47,
  serialized_end=98,
)


_DETECTEDOBJECT = _descriptor.Descriptor(
  name='DetectedObject',
  full_name='DetectedObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_name', full_name='DetectedObject.class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_idx', full_name='DetectedObject.class_idx', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='p1', full_name='DetectedObject.p1', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='p2', full_name='DetectedObject.p2', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='conf', full_name='DetectedObject.conf', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=100,
  serialized_end=209,
)


_POINT = _descriptor.Descriptor(
  name='Point',
  full_name='Point',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='Point.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='Point.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=211,
  serialized_end=240,
)

_DETECTEDOBJECTS.fields_by_name['objects'].message_type = _DETECTEDOBJECT
_DETECTEDOBJECT.fields_by_name['p1'].message_type = _POINT
_DETECTEDOBJECT.fields_by_name['p2'].message_type = _POINT
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['DetectedObjects'] = _DETECTEDOBJECTS
DESCRIPTOR.message_types_by_name['DetectedObject'] = _DETECTEDOBJECT
DESCRIPTOR.message_types_by_name['Point'] = _POINT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(
  DESCRIPTOR = _IMAGE,
  __module__ = 'yolov5_service_pb2'
  # @@protoc_insertion_point(class_scope:Image)
  ))
_sym_db.RegisterMessage(Image)

DetectedObjects = _reflection.GeneratedProtocolMessageType('DetectedObjects', (_message.Message,), dict(
  DESCRIPTOR = _DETECTEDOBJECTS,
  __module__ = 'yolov5_service_pb2'
  # @@protoc_insertion_point(class_scope:DetectedObjects)
  ))
_sym_db.RegisterMessage(DetectedObjects)

DetectedObject = _reflection.GeneratedProtocolMessageType('DetectedObject', (_message.Message,), dict(
  DESCRIPTOR = _DETECTEDOBJECT,
  __module__ = 'yolov5_service_pb2'
  # @@protoc_insertion_point(class_scope:DetectedObject)
  ))
_sym_db.RegisterMessage(DetectedObject)

Point = _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), dict(
  DESCRIPTOR = _POINT,
  __module__ = 'yolov5_service_pb2'
  # @@protoc_insertion_point(class_scope:Point)
  ))
_sym_db.RegisterMessage(Point)



_YOLOV5 = _descriptor.ServiceDescriptor(
  name='YoloV5',
  full_name='YoloV5',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=242,
  serialized_end=286,
  methods=[
  _descriptor.MethodDescriptor(
    name='detect',
    full_name='YoloV5.detect',
    index=0,
    containing_service=None,
    input_type=_IMAGE,
    output_type=_DETECTEDOBJECTS,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_YOLOV5)

DESCRIPTOR.services_by_name['YoloV5'] = _YOLOV5

# @@protoc_insertion_point(module_scope)
