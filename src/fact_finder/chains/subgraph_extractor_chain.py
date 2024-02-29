from typing import Dict, Any, Optional, List

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun


class SubgraphExtractorChain(Chain):
    raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        pass

    @property
    def output_keys(self) -> List[str]:
        pass

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        pass
