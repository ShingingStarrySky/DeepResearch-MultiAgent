from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                self._driver.verify_connectivity()
                logger.info(f"Neo4j 连接成功: {self.uri}")
            except ImportError:
                logger.warning("neo4j 驱动未安装，使用内存模式。pip install neo4j")
                self._driver = None
            except Exception as e:
                logger.warning(f"Neo4j 连接失败: {e}，使用内存模式")
                self._driver = None
        return self._driver

    def execute_write(self, query: str, parameters: dict = None) -> list:
        driver = self.driver
        if driver is None:
            return []
        parameters = parameters or {}
        with driver.session() as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, parameters))
            )
            return result

    def execute_read(self, query: str, parameters: dict = None) -> list:
        driver = self.driver
        if driver is None:
            return []
        parameters = parameters or {}
        with driver.session() as session:
            result = session.execute_read(
                lambda tx: list(tx.run(query, parameters))
            )
            return result

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
