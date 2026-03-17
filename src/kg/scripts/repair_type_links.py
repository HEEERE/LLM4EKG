import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_db.graph_database import GraphDatabaseManager


def main():
    logging.basicConfig(level=logging.INFO)
    gm = GraphDatabaseManager()
    # 统计前
    before = gm.get_type_link_stats()
    logging.info(f"before: {before}")
    stats = gm.cleanup_wrong_type_links()
    logging.info(f"cleanup: loops={stats.get('loops')}, subject_is_type={stats.get('subject_is_type')}, target_not_type={stats.get('target_not_type')}")
    count = gm.backfill_type_links()
    logging.info(f"backfill: affected_relations={count}")
    # 统计后
    after = gm.get_type_link_stats()
    logging.info(f"after: {after}")
    gm.close()


if __name__ == "__main__":
    main()
