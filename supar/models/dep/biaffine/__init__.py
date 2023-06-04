# -*- coding: utf-8 -*-

from .model import BiaffineDependencyModel, BiaffineDependencyWAttentionsModel
from .parser import BiaffineDependencyParser, BiaffineDepParserWRelations

__all__ = ['BiaffineDependencyModel', 'BiaffineDependencyParser',
           'BiaffineDependencyWAttentionsModel', 'BiaffineDepParserWRelations']
