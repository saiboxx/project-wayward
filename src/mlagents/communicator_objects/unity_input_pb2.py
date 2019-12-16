# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlagents/envs/communicator_objects/unity_input.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from src.mlagents.communicator_objects import unity_rl_input_pb2 as mlagents_dot_envs_dot_communicator__objects_dot_unity__rl__input__pb2
from src.mlagents.communicator_objects import unity_rl_initialization_input_pb2 as mlagents_dot_envs_dot_communicator__objects_dot_unity__rl__initialization__input__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlagents/envs/communicator_objects/unity_input.proto',
  package='communicator_objects',
  syntax='proto3',
  serialized_pb=_b('\n4mlagents/envs/communicator_objects/unity_input.proto\x12\x14\x63ommunicator_objects\x1a\x37mlagents/envs/communicator_objects/unity_rl_input.proto\x1a\x46mlagents/envs/communicator_objects/unity_rl_initialization_input.proto\"\xa4\x01\n\x0fUnityInputProto\x12\x39\n\x08rl_input\x18\x01 \x01(\x0b\x32\'.communicator_objects.UnityRLInputProto\x12V\n\x17rl_initialization_input\x18\x02 \x01(\x0b\x32\x35.communicator_objects.UnityRLInitializationInputProtoB\x1f\xaa\x02\x1cMLAgents.CommunicatorObjectsb\x06proto3')
  ,
  dependencies=[mlagents_dot_envs_dot_communicator__objects_dot_unity__rl__input__pb2.DESCRIPTOR,mlagents_dot_envs_dot_communicator__objects_dot_unity__rl__initialization__input__pb2.DESCRIPTOR,])




_UNITYINPUTPROTO = _descriptor.Descriptor(
  name='UnityInputProto',
  full_name='communicator_objects.UnityInputProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rl_input', full_name='communicator_objects.UnityInputProto.rl_input', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rl_initialization_input', full_name='communicator_objects.UnityInputProto.rl_initialization_input', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=372,
)

_UNITYINPUTPROTO.fields_by_name['rl_input'].message_type = mlagents_dot_envs_dot_communicator__objects_dot_unity__rl__input__pb2._UNITYRLINPUTPROTO
_UNITYINPUTPROTO.fields_by_name['rl_initialization_input'].message_type = mlagents_dot_envs_dot_communicator__objects_dot_unity__rl__initialization__input__pb2._UNITYRLINITIALIZATIONINPUTPROTO
DESCRIPTOR.message_types_by_name['UnityInputProto'] = _UNITYINPUTPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UnityInputProto = _reflection.GeneratedProtocolMessageType('UnityInputProto', (_message.Message,), dict(
  DESCRIPTOR = _UNITYINPUTPROTO,
  __module__ = 'src.mlagents.communicator_objects.unity_input_pb2'
  # @@protoc_insertion_point(class_scope:communicator_objects.UnityInputProto)
  ))
_sym_db.RegisterMessage(UnityInputProto)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\252\002\034MLAgents.CommunicatorObjects'))
# @@protoc_insertion_point(module_scope)
