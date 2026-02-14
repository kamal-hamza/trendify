"""
Management command for entity resolution using vector clustering and LLM.
Allows manual testing and running of entity resolution tasks.
"""

from django.core.management.base import BaseCommand, CommandError

from api.models import Entity, EntityNode, Topic, TopicEntityLink
from api.services.embeddings import get_hybrid_resolver, get_llm_resolver
from api.tasks import (
    calculate_entity_momentum,
    find_similar_entities,
    llm_entity_cleanup,
    resolve_topics_to_entities,
)


class Command(BaseCommand):
    help = "Resolve topics to entities using vector clustering and optional LLM"

    def add_arguments(self, parser):
        parser.add_argument(
            "--action",
            type=str,
            default="resolve",
            choices=["resolve", "llm-cleanup", "find-similar", "momentum", "stats"],
            help="Action to perform (default: resolve)",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.85,
            help="Similarity threshold for vector clustering (0.0-1.0, default: 0.85)",
        )
        parser.add_argument(
            "--entity-id",
            type=int,
            help="Entity ID for find-similar or momentum actions",
        )
        parser.add_argument(
            "--max-keywords",
            type=int,
            default=50,
            help="Maximum keywords for LLM cleanup (default: 50)",
        )
        parser.add_argument(
            "--async",
            action="store_true",
            dest="use_async",
            help="Run as async Celery task",
        )
        parser.add_argument(
            "--test-llm",
            action="store_true",
            help="Test if LLM (Ollama) is available",
        )

    def handle(self, *args, **options):
        action = options["action"]
        threshold = options["threshold"]
        entity_id = options.get("entity_id")
        max_keywords = options["max_keywords"]
        use_async = options["use_async"]
        test_llm = options["test_llm"]

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("TRENDIFY ENTITY RESOLUTION")
        self.stdout.write("=" * 70)

        # Test LLM availability if requested
        if test_llm:
            self._test_llm()
            return

        if action == "resolve":
            self._handle_resolve(threshold, use_async)
        elif action == "llm-cleanup":
            self._handle_llm_cleanup(max_keywords, use_async)
        elif action == "find-similar":
            self._handle_find_similar(entity_id, threshold, use_async)
        elif action == "momentum":
            self._handle_momentum(entity_id, use_async)
        elif action == "stats":
            self._handle_stats()

    def _test_llm(self):
        """Test if Ollama LLM is available"""
        self.stdout.write("\nTesting LLM availability...")

        llm_resolver = get_llm_resolver()
        is_available = llm_resolver.is_available()

        if is_available:
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n[SUCCESS] Ollama is running at {llm_resolver.base_url}"
                )
            )
            self.stdout.write(f"Model: {llm_resolver.model}")
            self.stdout.write("\nLLM-based entity resolution is enabled.")
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"\n[WARNING] Ollama not available at {llm_resolver.base_url}"
                )
            )
            self.stdout.write("\nTo enable LLM resolution:")
            self.stdout.write("  1. Install Ollama: https://ollama.ai/")
            self.stdout.write(f"  2. Run: ollama pull {llm_resolver.model}")
            self.stdout.write("  3. Ensure Ollama is running: ollama serve")
            self.stdout.write("\nVector-based resolution will still work without LLM.")

    def _handle_resolve(self, threshold, use_async):
        """Resolve topics to entities using vector clustering"""
        unlinked_count = Topic.objects.filter(
            is_active=True, entity_links__isnull=True
        ).count()

        self.stdout.write(f"\nFound {unlinked_count} unlinked topics")
        self.stdout.write(f"Similarity threshold: {threshold}")

        if unlinked_count == 0:
            self.stdout.write(self.style.SUCCESS("\nAll topics are already linked!"))
            return

        if use_async:
            self.stdout.write("\nQueuing resolution task...")
            from api.tasks import resolve_topics_to_entities as task

            result = task.delay(similarity_threshold=threshold)
            self.stdout.write(
                self.style.SUCCESS(f"[SUCCESS] Task queued! Task ID: {result.id}")
            )
        else:
            self.stdout.write("\nResolving topics (this may take a minute)...\n")

            result = resolve_topics_to_entities(similarity_threshold=threshold)

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.SUCCESS("RESOLUTION COMPLETE"))
            self.stdout.write("=" * 70)
            self.stdout.write(f"\nResults:")
            self.stdout.write(f"  - Topics processed: {result.get('topics_processed', 0)}")
            self.stdout.write(f"  - Entities created: {result.get('entities_created', 0)}")
            self.stdout.write(f"  - Links created: {result.get('links_created', 0)}")
            self.stdout.write(
                f"  - Resolution groups: {result.get('resolution_groups', 0)}"
            )

            # Show some examples
            entities = Entity.objects.all()[:5]
            if entities:
                self.stdout.write("\nSample entities created:")
                for entity in entities:
                    node_count = entity.nodes.count()
                    self.stdout.write(
                        f"  - {entity.canonical_name} ({node_count} variations)"
                    )

    def _handle_llm_cleanup(self, max_keywords, use_async):
        """Run LLM-based entity cleanup"""
        self.stdout.write(f"\nRunning LLM cleanup (max keywords: {max_keywords})...")

        # Test LLM availability first
        llm_resolver = get_llm_resolver()
        if not llm_resolver.is_available():
            self.stdout.write(
                self.style.WARNING(
                    "\n[WARNING] Ollama not available. Cannot run LLM cleanup."
                )
            )
            self.stdout.write("Run with --test-llm to see setup instructions.")
            return

        if use_async:
            from api.tasks import llm_entity_cleanup as task

            result = task.delay(max_keywords=max_keywords)
            self.stdout.write(
                self.style.SUCCESS(f"[SUCCESS] Task queued! Task ID: {result.id}")
            )
        else:
            self.stdout.write("\nProcessing with LLM (this may take a minute)...\n")

            result = llm_entity_cleanup(max_keywords=max_keywords)

            self.stdout.write(
                self.style.SUCCESS(
                    f"\n[SUCCESS] LLM cleanup complete! Resolved {result} entities."
                )
            )

    def _handle_find_similar(self, entity_id, threshold, use_async):
        """Find similar entities"""
        if not entity_id:
            raise CommandError("--entity-id is required for find-similar action")

        try:
            entity = Entity.objects.get(id=entity_id)
            self.stdout.write(f"\nFinding entities similar to: {entity.canonical_name}")
            self.stdout.write(f"Similarity threshold: {threshold}")

            if use_async:
                from api.tasks import find_similar_entities as task

                result = task.delay(entity_id=entity_id, similarity_threshold=threshold)
                self.stdout.write(
                    self.style.SUCCESS(f"[SUCCESS] Task queued! Task ID: {result.id}")
                )
            else:
                result = find_similar_entities(
                    entity_id=entity_id, similarity_threshold=threshold
                )

                if result:
                    self.stdout.write(f"\nFound {len(result)} similar entities:")
                    for item in result[:10]:
                        self.stdout.write(
                            f"  - {item['name']}: {item['similarity']:.3f}"
                        )
                else:
                    self.stdout.write("\nNo similar entities found.")

        except Entity.DoesNotExist:
            raise CommandError(f"Entity with ID {entity_id} not found")

    def _handle_momentum(self, entity_id, use_async):
        """Calculate entity momentum"""
        if not entity_id:
            raise CommandError("--entity-id is required for momentum action")

        try:
            entity = Entity.objects.get(id=entity_id)
            self.stdout.write(f"\nCalculating momentum for: {entity.canonical_name}")

            if use_async:
                from api.tasks import calculate_entity_momentum as task

                result = task.delay(entity_id=entity_id)
                self.stdout.write(
                    self.style.SUCCESS(f"[SUCCESS] Task queued! Task ID: {result.id}")
                )
            else:
                result = calculate_entity_momentum(entity_id=entity_id)

                if result:
                    self.stdout.write("\n" + "=" * 70)
                    self.stdout.write("ENTITY MOMENTUM")
                    self.stdout.write("=" * 70)
                    self.stdout.write(f"\nEntity: {result['entity']}")
                    self.stdout.write(f"Date: {result.get('date', 'N/A')}")
                    self.stdout.write(f"Total mentions: {result['total_mentions']}")
                    self.stdout.write(f"Total engagement: {result['total_engagement']}")
                    self.stdout.write(f"Average momentum: {result['avg_momentum']:.2f}")
                    self.stdout.write(f"Topics aggregated: {result['topics_count']}")
                else:
                    self.stdout.write("\nNo momentum data available.")

        except Entity.DoesNotExist:
            raise CommandError(f"Entity with ID {entity_id} not found")

    def _handle_stats(self):
        """Display entity resolution statistics"""
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("ENTITY RESOLUTION STATISTICS")
        self.stdout.write("=" * 70)

        total_topics = Topic.objects.filter(is_active=True).count()
        linked_topics = TopicEntityLink.objects.values("topic").distinct().count()
        unlinked_topics = total_topics - linked_topics

        total_entities = Entity.objects.filter(is_active=True).count()
        total_nodes = EntityNode.objects.count()

        self.stdout.write(f"\nTopics:")
        self.stdout.write(f"  - Total active: {total_topics}")
        self.stdout.write(f"  - Linked to entities: {linked_topics}")
        self.stdout.write(f"  - Unlinked: {unlinked_topics}")

        self.stdout.write(f"\nEntities:")
        self.stdout.write(f"  - Total entities: {total_entities}")
        self.stdout.write(f"  - Total entity nodes: {total_nodes}")

        # Resolution method breakdown
        vector_links = TopicEntityLink.objects.filter(resolution_method="VECTOR").count()
        llm_links = TopicEntityLink.objects.filter(resolution_method="LLM").count()
        exact_links = TopicEntityLink.objects.filter(resolution_method="EXACT").count()

        self.stdout.write(f"\nResolution methods:")
        self.stdout.write(f"  - Exact match: {exact_links}")
        self.stdout.write(f"  - Vector clustering: {vector_links}")
        self.stdout.write(f"  - LLM resolution: {llm_links}")

        # Top entities by topic count
        self.stdout.write(f"\nTop 10 entities by topic count:")
        from django.db.models import Count

        top_entities = (
            Entity.objects.annotate(
                topic_count=Count("topic_links") + Count("nodes__topic_links")
            )
            .order_by("-topic_count")[:10]
        )

        for entity in top_entities:
            self.stdout.write(f"  - {entity.canonical_name}: {entity.topic_count} topics")

        # LLM availability
        self.stdout.write(f"\nLLM Status:")
        llm_resolver = get_llm_resolver()
        if llm_resolver.is_available():
            self.stdout.write(
                self.style.SUCCESS(f"  - Ollama: Available ({llm_resolver.model})")
            )
        else:
            self.stdout.write(self.style.WARNING("  - Ollama: Not available"))

        self.stdout.write("\n" + "=" * 70)