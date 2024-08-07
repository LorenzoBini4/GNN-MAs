#!/user/bin/env python3
# map used to trace generated logs, 
malog = {
    'GT': {
        'ZINC': {
            # logs in ../graphtransformer ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'train': [
                                    # '../graphtransformer/out/ZINC_sparse_NoPE_BN/logs/GraphTransformer_ZINC_GPU0_18h24m58s_on_May_14_2024-use_bias_false/RUN_0/malog.pkl'
                                    # NOTE: the previous line is an example, replace with your logs
                                ],
                                'base': [
                                ]
                            },
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'train': [
                                ]
                            }
                        }
                    },
                },
            },
        },
        'TOX21': {
            # logs in ../graphtransformer ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'train': [
                                ],
                                'base': [
                                ]
                            },
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'train': [
                                ]
                            }
                        }
                    }
                }
            }
        },
        'PROTEINS': {
            # logs in ../gnn-lspe ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'train': [
                                ],
                                'base': [
                                ]
                            }
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'train': [
                                ]
                            }
                        }
                    }
                }
            }
        },
    },
    'SAN': {
        'ZINC': {
            # logs in ../SAN ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ],
                                    'base': [
                                    ]
                                }
                            }
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
        'TOX21': {
            # logs in ../gnn-lspe ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ],
                                    'base': [
                                    ]
                                }
                            }
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
        'PROTEINS': {
            # logs in ../gnn-lspe ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ],
                                    'base': [
                                    ]
                                }
                            }
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    'GraphiT': {
        'ZINC': {
            # logs in ../gnn-lspe ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ],
                                    'base': [
                                    ]
                                }
                            }
                        },
                        'ExplicitBias':{
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
        'TOX21': {
            # logs in ../gnn-lspe ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ],
                                    'base': [
                                    ]
                                }
                            }
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
        'PROTEINS': {
            # logs in ../gnn-lspe ...
            'sparse': {
                'NoPE': {
                    'BN': {
                        'NoBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ],
                                    'base': [
                                    ]
                                }
                            }
                        },
                        'ExplicitBias': {
                            'Edge': {
                                'Escore': {
                                    'train': [
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
