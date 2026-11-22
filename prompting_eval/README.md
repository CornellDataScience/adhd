# prompting_eval

Prompt building, citation mapping, and the 50-question eval set (steps 8 and 10).

`prompt.build_prompt(retrieve_response)` turns a retrieve-contract response into the
numbered-source string used by citation SFT and (later) `/generate`.
