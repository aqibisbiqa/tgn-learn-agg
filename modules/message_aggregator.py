from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device, memory):
    super(MeanMessageAggregator, self).__init__(device)
    self.memory = memory

  def aggregate(self, node_ids, messages):
    """Take mean of messages to each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    memory = self.memory.get_memory(unique_node_ids, "agg")
    # print(f"(agg) shape of memory is {memory.shape}")
    # print(f"(agg) max node id is {np.max(unique_node_ids)}")
    # print(f"(agg) tot node id is {len(unique_node_ids)}")

    # if len(unique_node_ids) < 150:
    #   print(unique_node_ids)

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

class AttnMessageAggregator(MessageAggregator):
  def __init__(self, device, memory, message_dimension, memory_dimension, num_heads, dropout):
    super(AttnMessageAggregator, self).__init__(device)
    self.memory = memory
    self.message_dimension = message_dimension
    self.memory_dimension = memory_dimension
    self.attention = nn.MultiheadAttention(
      embed_dim=memory_dimension,
      kdim=message_dimension,
      vdim=message_dimension,
      num_heads=num_heads,
      dropout=dropout,
    )
    self.mem_to_msg = nn.Linear(memory_dimension, message_dimension)

  def aggregate(self, node_ids, messages):
    """Attend to important messages for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    
    memory = self.memory.get_memory(unique_node_ids, "agg")
    # print(f"(agg) shape of memory is {memory.shape}")
    # print(f"(agg) max node id is {np.max(unique_node_ids)}")
    # print(f"(agg) tot node id is {len(unique_node_ids)}")

    for row_in_mem, node_id in enumerate(unique_node_ids):
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        curr_state = torch.unsqueeze(memory[row_in_mem], dim=0)
        curr_msgs = torch.stack([m[0] for m in messages[node_id]])
        agg_msg, _ = self.attention(
          query=curr_state,
          key=curr_msgs,
          value=curr_msgs,
          need_weights=False,
        )
        agg_msg = torch.squeeze(agg_msg, dim=0)
        agg_msg = self.mem_to_msg(agg_msg)
        # avg_msg = torch.mean(curr_msgs, dim=0)
        # lst_msg = messages[node_id][-1][0]
        unique_messages.append(agg_msg)
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

def get_message_aggregator(aggregator_type, device, memory, message_dimension, memory_dimension, num_heads=2, dropout=0.1):
  # match aggregator_type:
  #   case "last":
  #     return LastMessageAggregator(device=device)
  #   case "mean":
  #     return MeanMessageAggregator(device=device)
  #   case "attn":
  #     return AttnMessageAggregator(device=device,
  #                                  memory=memory,
  #                                  message_dimension=message_dimension, 
  #                                  memory_dimension=memory_dimension, 
  #                                  num_heads=num_heads, 
  #                                  dropout=dropout
  #                                 )
  #   case _:
  #     raise ValueError("Message aggregator {} not implemented".format(aggregator_type))

  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device, memory=memory)
  elif aggregator_type == "attn":
    return AttnMessageAggregator(device=device,
                                   memory=memory,
                                   message_dimension=message_dimension, 
                                   memory_dimension=memory_dimension, 
                                   num_heads=num_heads, 
                                   dropout=dropout
                                  )
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
