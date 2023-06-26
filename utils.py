def remove_option(parser, option):
    for action in parser._actions:
        if vars(action)['option_strings'][0] == option:
            parser._handle_conflict_resolve(None, [(option, action)])
            break
