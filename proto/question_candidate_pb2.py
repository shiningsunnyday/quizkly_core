# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/question_candidate.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/question_candidate.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1eproto/question_candidate.proto\"\x97\x01\n\x11QuestionCandidate\x12\x19\n\x11question_sentence\x18\x01 \x01(\t\x12\x1c\n\x0egap_candidates\x18\x02 \x03(\x0b\x32\x04.Gap\x12 \n\x0b\x64istractors\x18\x03 \x03(\x0b\x32\x0b.Distractor\x12\x11\n\x03gap\x18\x04 \x01(\x0b\x32\x04.Gap\x12\x14\n\x0c\x63ontext_text\x18\x05 \x01(\t\"\xf2\x01\n\x03Gap\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x13\n\x0bstart_index\x18\x02 \x01(\x05\x12\x11\n\tend_index\x18\x03 \x01(\x05\x12%\n\x0fpredicted_label\x18\x04 \x01(\x0e\x32\x0c.Gap.GapType\x12\x12\n\nconfidence\x18\x05 \x01(\x02\x12!\n\x0btrain_label\x18\x06 \x01(\x0e\x32\x0c.Gap.GapType\x12\x11\n\tembedding\x18\x07 \x03(\x02\x12\x10\n\x08pos_tags\x18\x08 \x03(\t\"2\n\x07GapType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0c\n\x08NEGATIVE\x10\x01\x12\x0c\n\x08POSITIVE\x10\x02\"\x1a\n\nDistractor\x12\x0c\n\x04text\x18\x01 \x01(\tb\x06proto3')
)



_GAP_GAPTYPE = _descriptor.EnumDescriptor(
  name='GapType',
  full_name='Gap.GapType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NEGATIVE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='POSITIVE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=381,
  serialized_end=431,
)
_sym_db.RegisterEnumDescriptor(_GAP_GAPTYPE)


_QUESTIONCANDIDATE = _descriptor.Descriptor(
  name='QuestionCandidate',
  full_name='QuestionCandidate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='question_sentence', full_name='QuestionCandidate.question_sentence', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gap_candidates', full_name='QuestionCandidate.gap_candidates', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='distractors', full_name='QuestionCandidate.distractors', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gap', full_name='QuestionCandidate.gap', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='context_text', full_name='QuestionCandidate.context_text', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=35,
  serialized_end=186,
)


_GAP = _descriptor.Descriptor(
  name='Gap',
  full_name='Gap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='Gap.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_index', full_name='Gap.start_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_index', full_name='Gap.end_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='predicted_label', full_name='Gap.predicted_label', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='Gap.confidence', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_label', full_name='Gap.train_label', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='embedding', full_name='Gap.embedding', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_tags', full_name='Gap.pos_tags', index=7,
      number=8, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _GAP_GAPTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=189,
  serialized_end=431,
)


_DISTRACTOR = _descriptor.Descriptor(
  name='Distractor',
  full_name='Distractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='Distractor.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=433,
  serialized_end=459,
)

_QUESTIONCANDIDATE.fields_by_name['gap_candidates'].message_type = _GAP
_QUESTIONCANDIDATE.fields_by_name['distractors'].message_type = _DISTRACTOR
_QUESTIONCANDIDATE.fields_by_name['gap'].message_type = _GAP
_GAP.fields_by_name['predicted_label'].enum_type = _GAP_GAPTYPE
_GAP.fields_by_name['train_label'].enum_type = _GAP_GAPTYPE
_GAP_GAPTYPE.containing_type = _GAP
DESCRIPTOR.message_types_by_name['QuestionCandidate'] = _QUESTIONCANDIDATE
DESCRIPTOR.message_types_by_name['Gap'] = _GAP
DESCRIPTOR.message_types_by_name['Distractor'] = _DISTRACTOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

QuestionCandidate = _reflection.GeneratedProtocolMessageType('QuestionCandidate', (_message.Message,), dict(
  DESCRIPTOR = _QUESTIONCANDIDATE,
  __module__ = 'proto.question_candidate_pb2'
  # @@protoc_insertion_point(class_scope:QuestionCandidate)
  ))
_sym_db.RegisterMessage(QuestionCandidate)

Gap = _reflection.GeneratedProtocolMessageType('Gap', (_message.Message,), dict(
  DESCRIPTOR = _GAP,
  __module__ = 'proto.question_candidate_pb2'
  # @@protoc_insertion_point(class_scope:Gap)
  ))
_sym_db.RegisterMessage(Gap)

Distractor = _reflection.GeneratedProtocolMessageType('Distractor', (_message.Message,), dict(
  DESCRIPTOR = _DISTRACTOR,
  __module__ = 'proto.question_candidate_pb2'
  # @@protoc_insertion_point(class_scope:Distractor)
  ))
_sym_db.RegisterMessage(Distractor)


# @@protoc_insertion_point(module_scope)
