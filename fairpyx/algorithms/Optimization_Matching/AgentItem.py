# """
#     "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

#     Programmer: Hadar Bitan, Yuval Ben-Simhon
#     Date: 19.5.2024
# """

# class AgentItem:
    
#     def __init__(self, str_format, int_format, matching) -> None:
#         self.str_format = str_format
#         self.int_format = int_format
#         self.matching = matching

#     def GetStrFormat(self):
#         return self.str_format
    
#     def GetStrFormatFromInt(self, int_format):

    
#     def GetIntFormat(self):
#         return self.int_format
    
#     def GetMatching(self):
#         return self.matching
    
#     def SetMatching(self, matching):
#         self.matching = matching

# # Helper function to create AgentItem instances and store them in a list
# def create_agent_items(prefix, start, end):
#     items = []
#     for i in range(start, end + 1):
#         str_format = f"{prefix}{i}"
#         int_format = i
#         matching = None  # or any initial value for matching
#         items.append(AgentItem(str_format, int_format, matching))
#     return items